import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

# Set the parameters for the RNN
input_size = 10  # number of input features
hidden_size = 20  # number of features in the hidden state
num_layers = 5  # number of recurrent layers

# Create a single RNN layer
rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True, dropout=0.5)

# Create dummy input data (sequence_length, batch_size, input_size)
sequence_length = 3
batch_size = 5
dummy_input = torch.randn(sequence_length, batch_size, input_size)

# Export the PyTorch model to ONNX format
onnx_model_path = "rnn.onnx"
torch.onnx.export(rnn, dummy_input, onnx_model_path, 
                  input_names=['input'], 
                  output_names=['output', 'hidden'])

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Check the model for correctness
onnx.checker.check_model(onnx_model)

# Initialize direction variable
direction = None

# Print out the shapes of tensors in the ONNX model
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
    
    # Extract and print the num_layers and direction attributes
    num_layers = next((attr.i for attr in node.attribute if attr.name == 'num_layers'), None)
    node_direction = next((attr.s for attr in node.attribute if attr.name == 'direction'), None)
    dropout = next((attr.f for attr in node.attribute if attr.name == 'dropout'), None)
    batch_first = next((attr.i for attr in node.attribute if attr.name == 'batch_first'), None)
    
    if num_layers is not None:
        print(f"Node {node.name}: num_layers = {num_layers}")
    if node_direction is not None:
        direction = node_direction.decode('utf-8')
        print(f"Node {node.name}: direction = {direction}")
    if dropout is not None:
        print(f"Node {node.name}: dropout = {dropout}")
    if batch_first is not None:
        print(f"Node {node.name}: batch_first = {batch_first}")

# Infer attributes from the hidden state shape
hidden_state_shape = [dim.dim_value for dim in onnx_model.graph.output[1].type.tensor_type.shape.dim]
is_bidirectional = direction == 'bidirectional'
print(f"Is bidirectional: {is_bidirectional}")
num_layers_inferred = hidden_state_shape[0] // (2 if is_bidirectional else 1)
batch_first = hidden_state_shape[1] == batch_size
if batch_first:
    batch_first = False
else:
    batch_first = True
print(f"Inferred num_layers: {num_layers_inferred}")
print(f"Inferred bidirectional: {is_bidirectional}")
print(f"Inferred batch_first: {batch_first}")