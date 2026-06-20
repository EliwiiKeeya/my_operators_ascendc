#!/bin/bash
${ASCEND_TOOLKIT_HOME}/python/site-packages/bin/msopgen gen \
    -i msOpGen_MyFusionCustom.json \
    -c ai_core-ascend910b1 \
    -lan cpp \
    -out MyFusionCustom
