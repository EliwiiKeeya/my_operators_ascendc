/****************************************************************************************************
 * File				: my_add_custom.cpp
 * Date				: 2025-03-04 23:21:39
 * Author			: Eliwii_Keeya
 * Description		: MyAddCustom kernel侧源文件
 * Last Modified	: 2025-03-04 23:21:39
 ****************************************************************************************************/
 #include "kernel_operator.h"
 constexpr int32_t BUFFER_NUM = 4; // tensor num for each queue
 
class KernelMyAdd {
public:
    __aicore__ inline KernelMyAdd() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR o, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        aGm.SetGlobalBuffer((__gm__ DTYPE_A *)a + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        bGm.SetGlobalBuffer((__gm__ DTYPE_B *)b + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        cGm.SetGlobalBuffer((__gm__ DTYPE_C *)c + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);        
        oGm.SetGlobalBuffer((__gm__ DTYPE_O *)o + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->tileLength * sizeof(DTYPE_A));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * sizeof(DTYPE_B));
        pipe.InitBuffer(inQueueC, BUFFER_NUM, this->tileLength * sizeof(DTYPE_C));
        pipe.InitBuffer(outQueueO, BUFFER_NUM, this->tileLength * sizeof(DTYPE_O));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.AllocTensor<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_B> bLocal = inQueueB.AllocTensor<DTYPE_B>();
        AscendC::LocalTensor<DTYPE_C> cLocal = inQueueC.AllocTensor<DTYPE_C>();
        AscendC::LocalTensor<half> tmpBufferLocal = tmpBuffer.AllocTensor<half>();

        AscendC::DataCopy(aLocal, aGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(bLocal, bGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(cLocal, cGm[progress * this->tileLength], this->tileLength);

        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
        inQueueC.EnQue(cLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.DeQue<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_B> bLocal = inQueueB.DeQue<DTYPE_B>();
        AscendC::LocalTensor<DTYPE_C> cLocal = inQueueC.DeQue<DTYPE_C>();
        AscendC::LocalTensor<DTYPE_O> oLocal = outQueueO.AllocTensor<DTYPE_O>();

        AscendC::LocalTensor<half> tmpTensorLocal = tmpBuffer.Get<half>();
        
        AscendC::Add(aLocal, bLocal, tmpTensorLocal, this->tileLength);
        AscendC::Add(cLocal, tmpTensorLocal, oLocal, this->tileLength);
        
        outQueueO.EnQue<DTYPE_O>(oLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueC.FreeTensor(cLocal);
        tmpBuffer.FreeTensor(tmpTensorLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_O> oLocal = outQueueO.DeQue<DTYPE_O>();
        AscendC::DataCopy(oGm[progress * this->tileLength], oLocal, this->tileLength);
        outQueueO.FreeTensor(oLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueB, inQueueC;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueO;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<DTYPE_A> aGm;
    AscendC::GlobalTensor<DTYPE_B> bGm;
    AscendC::GlobalTensor<DTYPE_C> cGm;
    AscendC::GlobalTensor<DTYPE_O> oGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};
 
extern "C" __global__ __aicore__ void my_add_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
 {
     GET_TILING_DATA(tiling_data, tiling);
     KernelMyAdd op;
    op.Init(a, b, c, o, tiling_data.totalLength, tiling_data.tileNum);
     op.Process();
 }
 
 #ifndef ASCENDC_CPU_DEBUG
 // call of kernel function
void my_add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                   uint8_t *o,  GM_ADDR workspace, uint8_t *tiling)
 {
    my_add_custom<<<blockDim, l2ctrl, stream>>>(a, b, c, o, tiling);
 }
 #endif
 