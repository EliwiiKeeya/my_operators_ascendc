/****************************************************************************************************
 * File				: my_add_custom.cpp
 * Date				: 2025-03-04 23:19:29
 * Author			: Eliwii_Keeya
 * Description		: MyAddCustom host侧源文件
 * Last Modified	: 2025-03-04 23:19:29
 ****************************************************************************************************/
#include "my_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
	TilingData tiling;
	uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
	context->SetBlockDim(BLOCK_DIM);
	tiling.set_totalLength(totalLength);
	tiling.set_tileNum(TILE_NUM);
	tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
						context->GetRawTilingData()->GetCapacity());
	context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
	size_t *currentWorkspace = context->GetWorkspaceSizes(1);
	currentWorkspace[0] = 0;
	return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static graphStatus InferShape(gert::InferShapeContext *context)
{
	const gert::Shape *x1_shape = context->GetInputShape(0);
	gert::Shape *y_shape = context->GetOutputShape(0);
	*y_shape = *x1_shape;
	return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
	const auto inputDataType = context->GetInputDataType(0);
	context->SetOutputDataType(0, inputDataType);
	return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops
{
class MyAddCustom : public OpDef
{
  public:
  explicit MyAddCustom(const char *name)
      : OpDef(name)
  {
    this->Input("a")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("b")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("c")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("o")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
  }
};
OP_ADD(MyAddCustom);
}  // namespace ops
