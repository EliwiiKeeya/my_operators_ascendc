#!/bin/bash
export DDK_PATH=${ASCEND_TOOLKIT_HOME}
export NPU_HOST_LIB=${ASCEND_TOOLKIT_HOME}/aarch64-linux/devlib
echo DDK_PATH: ${DDK_PATH}
echo NPU_HOST_LIB: ${NPU_HOST_LIB}
${ASCEND_TOOLKIT_HOME}/python/site-packages/bin/msopst run \
    -i msOpST_MyFusionCustom.json \
    -soc Ascend910B1 \
    -out MyFusionCustomST
