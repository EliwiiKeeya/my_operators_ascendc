/****************************************************************************************************
 * File				: my_matmul_custom_tiling.h
 * Date				: 2025-03-05 17:43:37
 * Author			: Eliwii_Keeya
 * Description		: MyMatmulCustom host侧头文件
 * Last Modified	: 2025-03-05 17:43:37
 ****************************************************************************************************/
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MyMatmulCustomTilingData)
	TILING_DATA_FIELD_DEF(uint32_t, localMemSize);
	TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MyMatmulCustom, MyMatmulCustomTilingData)
}
