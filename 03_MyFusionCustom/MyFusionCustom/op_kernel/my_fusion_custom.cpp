/****************************************************************************************************
 * File				: my_fusion_custom.cpp
 * Date				: 2025-03-14 08:55:34
 * Author			: Eliwii_Keeya
 * Description		: MyFusionCustom host侧源文件
 * Last Modified	: 2025-03-14 08:55:34
 ****************************************************************************************************/
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType> class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                uint64_t memSize, const TCubeTiling &tiling);
    template <bool setTmpSpace = false> __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                    int32_t &offsetC);
    
    using MatmulTypeA = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>;
    using MatmulTypeB = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>;
    using MatmulTypeC = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>;

    Matmul<MatmulTypeA, MatmulTypeB, MatmulTypeC> matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
};

/**
* @brief  Set matmul input and output gm addr of current core.
* @param  a: A matrix gm addr.
* @param  b: B matrix gm addr.
* @param  c: C matrix gm addr.
* @param  workspace: Temporary gm space addr required by matmul calc.
* @param  tiling: matmul tiling data.
* @retval None
*/
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                                        GM_ADDR workspace, uint64_t memSize, const TCubeTiling &tiling)
{
    this->tiling = tiling;
    this->localMemSize = memSize;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC); // Calculate the gm offset based on the blockidx.
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

/**
* @brief  Main process of matmul calculation.
* @param  pipe: Global memory and sync management TPipe object.
* @retval None
*/
template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if (GetBlockIdx() >= 1) {
        return;
    }
    // Set temp UB space if the setTmpSpace is true.
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, localMemSize);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(localMemSize);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}

/**
* @brief  Calculate the gm offset based on the blockidx.
* @param  blockIdx: Current Core blockidx.
* @param  tiling: Matmul tiling data.
* @param  offsetA: Gm offset of A matrix.
* @param  offsetB: Gm offset of B matrix.
* @param  offsetC: Gm offset of C matrix.
* @retval None
*/
template <typename aType, typename bType, typename cType>
__aicore__ inline void
MatmulKernel<aType, bType, cType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA,
                                                        int32_t &offsetB, int32_t &offsetC)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
}

/**
* @brief  matmul kernel function entry.
* @param  a: A matrix gm addr.
* @param  b: B matrix gm addr.
* @param  c: C matrix gm addr.
* @param  workspace: Temporary gm space addr required by matmul calc.
* @param  tiling: Tiling data addr. 
* @retval None
*/
extern "C" __global__ __aicore__ void my_fusion_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernel<half, half, float> matmulKernel;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData); // Initialize the matmul object.
    matmulKernel.Init(a, b, c, workspace, tilingData.localMemSize, tilingData.cubeTilingData);
    if (TILING_KEY_IS(1)) {
        matmulKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    }
}
