#include "kernel_operator.h"

extern "C" __global__ __aicore__ void my_add_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}