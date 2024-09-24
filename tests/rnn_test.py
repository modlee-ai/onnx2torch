import torch
import onnx
import numpy as np
from onnx2torch import convert
from utils.common import check_onnx_model

### we define the RNN model in torch 
# rnn = torch.nn.RNN(
#     input_size=15,
#     hidden_size=3,
#     num_layers=2,
#     bias=True,
#     batch_first=False,
#     dropout=0.5,
#     bidirectional=False,
# )

torch.device('cpu')

# rnn = torch.nn.LSTM(
#     input_size=3,
#     hidden_size=3,
#     num_layers=2,
#     bias=True,
#     batch_first=False,
#     dropout=0.5,
#     bidirectional=False,
# )


# rnn = torch.nn.GRU(
#     input_size=3,
#     hidden_size=3,
#     num_layers=2,
#     bias=True,
#     batch_first=False,
#     dropout=0.5,
#     bidirectional=False,
# )

class testModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=15,
            hidden_size=4,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.5,
            bidirectional=False
        )

    def forward(self, x):
        return self.rnn(x)


'''
for 2 num layers

 {'_proto': input: "input"
input: "onnx::RNN_57"
input: "onnx::RNN_58"
input: "onnx::RNN_59"
input: ""
input: "/rnn/Slice_output_0"
output: "/rnn/RNN_output_0"
output: "/rnn/RNN_output_1"
name: "/rnn/RNN"
op_type: "RNN"
attribute {
  name: "activations"
  strings: "Tanh"
  type: STRINGS
}
attribute {
  name: "hidden_size"
  i: 5
  type: INT
}
, '_unique_name': 'rnn/RNN', '_input_values': ('input', 'onnx::RNN_57', 'onnx::RNN_58', 'onnx::RNN_59', '', '/rnn/Slice_output_0'), '_output_values': ('/rnn/RNN_output_0', '/rnn/RNN_output_1'), '_inputs': None, '_proto_attributes': {'activations': ['Tanh'], 'hidden_size': 5}}
'''

'''
for 3 layers

Input 1:  torch.Size([1, 5, 3])
Input 2:  torch.Size([1, 5, 5])
Input 3:  torch.Size([1, 10])

####################

{'_proto': input: "input"
input: "onnx::RNN_76"
input: "onnx::RNN_77"
input: "onnx::RNN_78"
input: ""
input: "/rnn/Slice_output_0"
output: "/rnn/RNN_output_0"
output: "/rnn/RNN_output_1"
name: "/rnn/RNN"
op_type: "RNN"
attribute {
  name: "activations"
  strings: "Tanh"
  type: STRINGS
}
attribute {
  name: "hidden_size"
  i: 5
  type: INT
}
, '_unique_name': 'rnn/RNN', '_input_values': ('input', 'onnx::RNN_76', 'onnx::RNN_77', 'onnx::RNN_78', '', '/rnn/Slice_output_0'), '_output_values': ('/rnn/RNN_output_0', '/rnn/RNN_output_1'), '_inputs': None, '_proto_attributes': {'activations': ['Tanh'], 'hidden_size': 5}}
'''


rnn = testModel()
rnn.eval()


dummy_input = torch.randn(10, 5, 15)

# Ensure dummy_input is a tensor
if not isinstance(dummy_input, torch.Tensor):
    raise TypeError(f"Expected dummy_input to be a tensor, but got {type(dummy_input)}")

onnx_model_path = "rnn.onnx"
# torch.onnx.export(
#     rnn,
#     dummy_input,
#     onnx_model_path,
#     opset_version=14,
#     export_params=True,
#     do_constant_folding=True,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}},
# )

torch.onnx.export(rnn, dummy_input, onnx_model_path, 
                  input_names=['input'], 
                  output_names=['output', 'hidden'])

# 
# print("Model output pre conversion shape: ", output.shape)

# Inspect and print the shapes of all tensors in the RNN's state_dict


# Load the ONNX model and convert it
# Load the ONNX model and convert it to PyTorch
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)


for node in onnx_model.graph.node:
    for input_name in node.input:
        input_info = next((val for val in onnx_model.graph.input if val.name == input_name), None)
        if input_info:
            input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
            print(f"Input {input_name}: shape = {input_shape}")
    
    for output_name in node.output:
        output_info = next((val for val in onnx_model.graph.output if val.name == output_name), None)
        if output_info:
            output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
            print(f"Output {output_name}: shape = {output_shape}")

dummy_input_np = dummy_input.numpy()
output, hidden = rnn.forward(dummy_input)
state_dict = rnn.state_dict()
for name, tensor in state_dict.items():
    print(f"{name}: {tensor.shape}")
#breakpoint()

import onnxruntime
try:
    check_onnx_model(
        onnx_model,
        {"input": dummy_input_np},
        atol_onnx_torch=10**-4,  # Increase tolerance
        atol_torch_cpu_cuda=10**-5,  # Increase tolerance
    )
except AssertionError as e:
    print("AssertionError: ort and torch outputs have significant difference")
    # Compare the outputs
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    torch_output = rnn.forward(dummy_input)[0].detach().numpy()
    
    print("ORT output:", ort_outputs)
    print("Torch output:", torch_output)
    
    # Print the differences
    for ort_output, torch_out in zip(ort_outputs, torch_output):
        print("Difference:", np.abs(ort_output - torch_out))

'''
1) get torch model as code
2) add breakpoints to figure out the types of the following input arguments
gru = self.GRU(transpose, torch.tenser(initializers_onyx_initializer, initializers_on_it_initializer,))
3) modify code until the inputs pass through the GRU layer
4) create function that applies necessary modifications within the onyx to torch library
'''

