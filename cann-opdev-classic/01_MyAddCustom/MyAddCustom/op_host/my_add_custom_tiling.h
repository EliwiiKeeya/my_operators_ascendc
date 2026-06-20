/****************************************************************************************************
 * File				: my_add_custom_tiling.h
 * Date				: 2025-03-04 23:18:41
 * Author			: Eliwii_Keeya
 * Description		: MyAddCustom host侧头文件
 * Last Modified	: 2025-03-04 23:18:41
 ****************************************************************************************************/
#ifndef MY_ADD_CUSTOM_TILING_H
#define MY_ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData);
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MyAddCustom, TilingData);
} // namespace optiling
#endif // ADD_CUSTOM_TILING_H
