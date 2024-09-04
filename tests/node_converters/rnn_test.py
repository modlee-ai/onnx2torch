import torch
import onnx
import numpy as np
from onnx2torch import convert
from torch.fx import symbolic_trace


### we define the RNN model in torch 
# rnn = torch.nn.RNN(
#     input_size=10,
#     hidden_size=20,
#     num_layers=1,
#     bias=True,
#     batch_first=False,
#     dropout=0.0,
#     bidirectional=False,
# )

torch.device('cpu')

# rnn = torch.nn.LSTM(
#     input_size=10,
#     hidden_size=20,
#     num_layers=1,
#     bias=True,
#     batch_first=False,
#     dropout=0.5,
#     bidirectional=False,
# )


# rnn = torch.nn.GRU(
#     input_size=10,
#     hidden_size=5,
#     num_layers=2,
#     bias=True,
#     batch_first=True,
#     dropout=0.5,
#     bidirectional=False,
# )

class testModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=10,
            hidden_size=5,
            num_layers=2,
            bias=True,
            batch_first=False,
            dropout=0.5,
            bidirectional=False
        )

    def forward(self, x):
        return self.rnn(x)


rnn = testModel()
rnn.eval()


dummy_input = torch.randn(3, 5, 10)
onnx_model_path = "rnn.onnx"
torch.onnx.export(
    rnn,
    dummy_input,
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}},
)



output, hidden = rnn.forward(dummy_input)
print("Model output pre conversion shape: ", output.shape)
# Load the ONNX model and convert it
# Load the ONNX model and convert it to PyTorch
model = onnx.load(onnx_model_path)
onnx.checker.check_model(model)

converted_model = convert(model)

def fix_device_issues(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Module):
            fix_device_issues(child)
        if isinstance(child, torch.Tensor):
            # Ensure the device is correctly set
            if not isinstance(child.device, torch.device):
                # Set the device to the appropriate one (e.g., cpu or cuda)
                child = child.to(torch.device('cpu'))
            setattr(module, name, child)
    return module

# Apply this function to the entire converted model
# Apply the function to the entire converted model
#converted_model = fix_device_issues(converted_model)
#breakpoint()
code = converted_model.code
#print(code)
filename = "rnnModel.py"
with open(filename, 'w') as f:
    f.write(code)
    f.close()

try:
    # Forward pass with dummy data
    output, hidden = converted_model(dummy_input)
    print("Model output:", output)
    print("Output shape of converted model: ", output.shape)
except Exception as e:
    print(f"Error during model inference: {e}")

'''
1) get torch model as code
2) add breakpoints to figure out the types of the following input arguments
gru = self.GRU(transpose, torch.tenser(initializers_onyx_initializer, initializers_on_it_initializer,))
3) modify code until the inputs pass through the GRU layer
4) create function that applies necessary modifications within the onyx to torch library
'''

