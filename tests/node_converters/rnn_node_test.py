import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# Define the RNN parameters
input_size = 10
hidden_size = 20
batch_size = 3
sequence_length = 5
num_layers = 9  # Number of RNN layers

# Create input tensors for the RNN
X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [sequence_length, batch_size, input_size])
W = helper.make_tensor_value_info('W', onnx.TensorProto.FLOAT, [num_layers, hidden_size, input_size])
R = helper.make_tensor_value_info('R', onnx.TensorProto.FLOAT, [num_layers, hidden_size, hidden_size])
B = helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [num_layers, 2 * hidden_size])
H0 = helper.make_tensor_value_info('H0', onnx.TensorProto.FLOAT, [num_layers, batch_size, hidden_size])
Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [sequence_length, num_layers, batch_size, hidden_size])
Y_h = helper.make_tensor_value_info('Y_h', onnx.TensorProto.FLOAT, [num_layers, batch_size, hidden_size])

# Create the RNN node
rnn_node = helper.make_node(
    'RNN',
    inputs=['X', 'W', 'R', 'B', 'H0'],
    outputs=['Y', 'Y_h'],
    hidden_size=hidden_size,
    direction='forward',  # You can also set 'bidirectional' if needed
)

# Create the ONNX graph, including value_info for inputs and outputs
graph_def = helper.make_graph(
    [rnn_node],
    'rnn-model',
    [X, W, R, B, H0],  # Inputs
    [Y, Y_h],  # Outputs
    value_info=[X, W, R, B, H0, Y, Y_h]  # Explicitly add value_info for inputs and outputs
)

# Create the ONNX model
model_def = helper.make_model(graph_def, producer_name='onnx-rnn')

# Generate random initial weights and inputs for multiple layers
W_data = np.random.randn(num_layers, hidden_size, input_size).astype(np.float32)
R_data = np.random.randn(num_layers, hidden_size, hidden_size).astype(np.float32)
B_data = np.random.randn(num_layers, 2 * hidden_size).astype(np.float32)
H0_data = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
X_data = np.random.randn(sequence_length, batch_size, input_size).astype(np.float32)

# Create the initializers (weights and biases)
W_init = numpy_helper.from_array(W_data, name='W')
R_init = numpy_helper.from_array(R_data, name='R')
B_init = numpy_helper.from_array(B_data, name='B')
H0_init = numpy_helper.from_array(H0_data, name='H0')

# Add the initializers to the model
model_def.graph.initializer.extend([W_init, R_init, B_init, H0_init])

# Save the model (optional)
onnx.save(model_def, 'rnn_model.onnx')

# Load the model to verify and inspect
model = onnx.load('rnn_model.onnx')

# Check the model
onnx.checker.check_model(model)

# Inspect the model nodes and tensor shapes
for node in model.graph.node:
    print(f"Node: {node.op_type}")
    for input_name in node.input:
        input_info = next((val for val in model.graph.value_info if val.name == input_name), None)
        if input_info:
            input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
            print(f"  Input {input_name}: shape = {input_shape}")
    for output_name in node.output:
        output_info = next((val for val in model.graph.value_info if val.name == output_name), None)
        if output_info:
            output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
            print(f"  Output {output_name}: shape = {output_shape}")
