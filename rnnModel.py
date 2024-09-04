


def forward(self, input_1):
    rnn_shape = getattr(self, "rnn/Shape")(input_1)
    rnn_constant = getattr(self, "rnn/Constant")()
    rnn_gather = getattr(self, "rnn/Gather")(rnn_shape, rnn_constant);  rnn_shape = rnn_constant = None
    rnn_constant_1 = getattr(self, "rnn/Constant_1")()
    constant_16 = self.Constant_16()
    rnn_unsqueeze = getattr(self, "rnn/Unsqueeze")(rnn_gather);  rnn_gather = None
    rnn_constant_2 = getattr(self, "rnn/Constant_2")()
    rnn_concat = getattr(self, "rnn/Concat")(rnn_constant_1, rnn_unsqueeze, rnn_constant_2);  rnn_constant_1 = rnn_unsqueeze = rnn_constant_2 = None
    rnn_constant_of_shape = getattr(self, "rnn/ConstantOfShape")(rnn_concat);  rnn_concat = None
    rnn_constant_3 = getattr(self, "rnn/Constant_3")()
    rnn_constant_4 = getattr(self, "rnn/Constant_4")()
    rnn_constant_5 = getattr(self, "rnn/Constant_5")()
    rnn_slice = getattr(self, "rnn/Slice")(rnn_constant_of_shape, rnn_constant_4, rnn_constant_5, rnn_constant_3);  rnn_constant_4 = rnn_constant_5 = rnn_constant_3 = None
    initializers_onnx_initializer_0 = self.initializers.onnx_initializer_0
    initializers_onnx_initializer_1 = self.initializers.onnx_initializer_1
    initializers_onnx_initializer_2 = self.initializers.onnx_initializer_2
    rnn_gru = getattr(self, "rnn/GRU")(input_1, initializers_onnx_initializer_0, initializers_onnx_initializer_1, initializers_onnx_initializer_2);  input_1 = initializers_onnx_initializer_0 = initializers_onnx_initializer_1 = initializers_onnx_initializer_2 = None
    rnn_constant_6 = getattr(self, "rnn/Constant_6")()
    getitem = rnn_gru[0]
    rnn_squeeze = getattr(self, "rnn/Squeeze")(getitem, rnn_constant_6);  getitem = rnn_constant_6 = None
    rnn_constant_7 = getattr(self, "rnn/Constant_7")()
    rnn_constant_8 = getattr(self, "rnn/Constant_8")()
    rnn_constant_9 = getattr(self, "rnn/Constant_9")()
    rnn_slice_1 = getattr(self, "rnn/Slice_1")(rnn_constant_of_shape, rnn_constant_8, rnn_constant_9, rnn_constant_7);  rnn_constant_of_shape = rnn_constant_8 = rnn_constant_9 = rnn_constant_7 = None
    initializers_onnx_initializer_3 = self.initializers.onnx_initializer_3
    initializers_onnx_initializer_4 = self.initializers.onnx_initializer_4
    initializers_onnx_initializer_5 = self.initializers.onnx_initializer_5
    rnn_gru_1 = getattr(self, "rnn/GRU_1")(rnn_squeeze, initializers_onnx_initializer_3, initializers_onnx_initializer_4, initializers_onnx_initializer_5);  rnn_squeeze = initializers_onnx_initializer_3 = initializers_onnx_initializer_4 = initializers_onnx_initializer_5 = None
    rnn_constant_10 = getattr(self, "rnn/Constant_10")()
    getitem_1 = rnn_gru_1[0]
    rnn_squeeze_1 = getattr(self, "rnn/Squeeze_1")(getitem_1, rnn_constant_10);  getitem_1 = rnn_constant_10 = None
    getitem_2 = rnn_gru[1];  rnn_gru = None
    getitem_3 = rnn_gru_1[1];  rnn_gru_1 = None
    rnn_concat_1 = getattr(self, "rnn/Concat_1")(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    return [rnn_squeeze_1, rnn_concat_1]
    