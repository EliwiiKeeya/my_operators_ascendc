import mindspore
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import DataType, CustomRegOp


class MyMatmulCustomNet(Cell):
    def __init__(self, func, out_shape):
        super(MyMatmulCustomNet, self).__init__()
        reg_info = CustomRegOp("MyMatmulCustom") \
            .input(0, "a", "required") \
            .input(1, "b", "required") \
            .output(0, "c", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()

        self.my_custom_add = ops.Custom(func=func, out_shape=out_shape, out_dtype=self.infer_dtype, func_type="aot", bprop=None,
                                        reg_info=reg_info)

    def construct(self, x, y):
        res = self.my_custom_add(x, y)
        return res

    @staticmethod
    def infer_dtype(arg0, arg1):
        return mindspore.float32

    @staticmethod
    def infer_shape(arg0, arg1):
        return (arg0[0], arg1[1])


mindspore.set_context(jit_config={"jit_level": "O0"})
mindspore.set_device("Ascend")

a = ops.ones([1024, 256], mindspore.float16)
b = ops.ones([256, 640], mindspore.float16)
c = ops.ones([640], mindspore.float32)

net = MyMatmulCustomNet("MyMatmulCustom", MyMatmulCustomNet.infer_shape)

print(net(a, b))
