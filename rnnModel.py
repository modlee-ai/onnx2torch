


def forward(self, input_1):
    shape = self.Shape(input_1)
    constant = self.Constant()
    gather = self.Gather(shape, constant);  shape = constant = None
    constant_1 = self.Constant_1()
    constant_9 = self.Constant_9()
    unsqueeze = self.Unsqueeze(gather);  gather = None
    constant_2 = self.Constant_2()
    concat = self.Concat(constant_1, unsqueeze, constant_2);  constant_1 = unsqueeze = constant_2 = None
    constant_of_shape = self.ConstantOfShape(concat);  concat = None
    initializers_onnx_initializer_0 = self.initializers.onnx_initializer_0
    initializers_onnx_initializer_1 = self.initializers.onnx_initializer_1
    initializers_onnx_initializer_2 = self.initializers.onnx_initializer_2
    rnn = self.RNN(input_1, initializers_onnx_initializer_0, initializers_onnx_initializer_1, initializers_onnx_initializer_2);  input_1 = initializers_onnx_initializer_0 = initializers_onnx_initializer_1 = initializers_onnx_initializer_2 = None
    constant_3 = self.Constant_3()
    getitem = rnn[0]
    squeeze = self.Squeeze(getitem, constant_3);  getitem = constant_3 = None
    return [squeeze, rnn]
    