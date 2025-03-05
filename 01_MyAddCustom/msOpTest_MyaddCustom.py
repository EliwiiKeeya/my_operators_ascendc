import mindspore
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import DataType, CustomRegOp


class MyAddCustomNet(Cell):
    def __init__(self, func, out_shape):
        super(MyAddCustomNet, self).__init__()
        reg_info = CustomRegOp("MyAddCustom") \
            .input(0, "a", "required") \
            .input(1, "b", "required") \
            .input(2, "c", "required") \
            .output(0, "o", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()

        self.my_custom_add = ops.Custom(func=func, out_shape=out_shape, out_dtype=self.infer_dtype, func_type="aot", bprop=None,
                                        reg_info=reg_info)

    def construct(self, x, y, z):
        res = self.my_custom_add(x, y, z)
        return res

    @staticmethod
    def infer_dtype(arg0, arg1, arg2):
        return arg0

    @staticmethod
    def infer_shape(arg0, arg1, arg2):
        return arg0


mindspore.set_context(jit_config={"jit_level": "O0"})
mindspore.set_device("Ascend")

a = ops.ones([8, 2048], mindspore.float32)
b = ops.ones([8, 2048], mindspore.float32)
c = ops.ones([8, 2048], mindspore.float32)

# 通过lambda实现infer shape函数
net = MyAddCustomNet("MyAddCustom", MyAddCustomNet.infer_shape)

print(net(a, b, c))
