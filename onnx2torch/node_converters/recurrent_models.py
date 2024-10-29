import torch
from torch import nn
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OperationConverterResult

_RNN_CLASS_FROM_TYPE = {
    'RNN': nn.RNN,
    'LSTM': nn.LSTM,
    'GRU': nn.GRU,
}

class RNNWrapper(nn.Module):
    def __init__(self, rnn_module, input_size, hidden_size, num_layers, batch_first, dropout, bidirectional):
        super(RNNWrapper, self).__init__()
        self.rnn_module = rnn_module
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self._printed_attributes = False

    def forward(self, input_1, *args):
        print(f"Initial input shape: {input_1.shape}")
        batch_size = input_1.shape[0] if self.batch_first else input_1.shape[1]
        seq_len = input_1.shape[1] if self.batch_first else input_1.shape[0]

        if not isinstance(input_1, torch.Tensor):
            raise TypeError(f"Expected input_1 to be a tensor, but got {type(input_1)}")

        if not self._printed_attributes:
            print(f"RNN Attributes - Input size: {self.input_size}, Hidden size: {self.hidden_size}, "
                  f"Num layers: {self.num_layers}, Bidirectional: {self.bidirectional}, "
                  f"Batch first: {self.batch_first}, Dropout: {self.dropout}")
            self._printed_attributes = True

        if self.batch_first:
            if input_1.dim() == 3:
                input_1 = input_1.permute(1, 0, 2)  # (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)

        # Check if the hidden state is passed
        if len(args) == 2:
            hx = args
            output, hidden = self.rnn_module(input_1, hx)
        else:
            output, hidden = self.rnn_module(input_1)

        print(f"Output shape before reshaping: {output.shape}")
        
        # If hidden is a tuple (LSTM), unpack it
        cell_state = None
        if isinstance(hidden, tuple):
            hidden_state, cell_state = hidden
            print(f"Hidden state shape: {hidden_state.shape}, Cell state shape: {cell_state.shape}")
        else:
            hidden_state = hidden  # For GRU or simple RNN

        if self.bidirectional:
            seq_len, batch_size, hidden_size_total = output.shape  # (seq_len, batch_size, hidden_size * 2)
            num_directions = 2
            hidden_size = hidden_size_total // num_directions

            # Reshape to (seq_len, batch_size, num_directions, hidden_size)
            output = output.view(seq_len, batch_size, num_directions, hidden_size)
            print(f"Output after reshaping for bidirectional: {output.shape}")

        # Handle output permutation back to (batch_size, seq_len, hidden_size) if batch_first=True
        if self.batch_first:
            if output.dim() == 4:
                output = output.permute(1, 0, 2, 3)  # (seq_len, batch_size, num_directions, hidden_size) to (batch_size, num_directions, seq_len, hidden_size)
                output = output.contiguous().view(batch_size, seq_len, -1)  # Reshape to (batch_size, seq_len, hidden_size * num_directions)
            elif output.dim() == 3:
                output = output.permute(1, 0, 2)  # From (seq_len, batch_size, hidden_size) back to (batch_size, seq_len, hidden_size)

        #return output, (hidden_state, cell_state) if isinstance(hidden, tuple) else hidden
        return output, hidden_state

@add_converter(operation_type='RNN', version=14)
@add_converter(operation_type='LSTM', version=14)
@add_converter(operation_type='GRU', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    rnn_type = node.operation_type
    weights_ih_name = node.input_values[1]
    weights_hh_name = node.input_values[2]
    
    if weights_ih_name not in graph.initializers or weights_hh_name not in graph.initializers:
        raise Exception(f"Graph does not have necessary weight tensors for {rnn_type}")

    # Convert tensors to nn.Parameter
    weights_ih = torch.nn.Parameter(graph.initializers[weights_ih_name].to_torch())
    weights_hh = torch.nn.Parameter(graph.initializers[weights_hh_name].to_torch())

    bias_ih = None
    bias_hh = None

    if len(node.input_values) > 3:
        bias_ih_name = node.input_values[3]
        bias_hh_name = node.input_values[4]
        if bias_ih_name in graph.initializers:
            bias_ih = torch.nn.Parameter(graph.initializers[bias_ih_name].to_torch())
        else:
            print(f"Warning: Bias tensor {bias_ih_name} not found in graph initializers.")
        
        if bias_hh_name in graph.initializers:
            bias_hh = torch.nn.Parameter(graph.initializers[bias_hh_name].to_torch())
        else:
            print(f"Warning: Bias tensor {bias_hh_name} not found in graph initializers.")

    # Extract or infer attributes
    input_size = node.attributes.get('input_size')
    hidden_size = node.attributes.get('hidden_size', None)
    num_layers = node.attributes.get('num_layers', 1)
    dropout = node.attributes.get('dropout', 0.0)
    batch_first = node.attributes.get('batch_first', None)
    
    # Extract direction attribute and set bidirectionality
    direction = node.attributes.get('direction', 'forward')
    if isinstance(direction, bytes):
        direction = direction.decode('utf-8')  # Decode if it's a bytes object
    is_bidirectional = direction == 'bidirectional'

    # If input_size is not provided, infer from the input tensor's shape
    input_shape = None
    if input_size is None and node.input_values:
        input_tensor_name = node.input_values[0]
        input_tensor_info = graph.value_info.get(input_tensor_name, None)
        
        if input_tensor_info:
            input_shape = [dim.dim_value for dim in input_tensor_info.type.tensor_type.shape.dim]
            if len(input_shape) == 3:
                input_size = input_shape[-1]
            else:
                raise ValueError(f"Unexpected input tensor shape {input_shape} for {rnn_type}")
        else:
            raise ValueError(f"Could not find input tensor information for {input_tensor_name}")

    if input_size is None or hidden_size is None:
        raise ValueError(f"Invalid RNN configuration: input_size and hidden_size must be defined (got input_size={input_size}, hidden_size={hidden_size})")

    # Retrieve output tensor shape (used to determine batch_first)
    if len(graph.proto.output) > 1:
        output_info = graph.proto.output[0]  # Output shape
        hidden_info = graph.proto.output[1]  # Hidden state shape

        if output_info and output_info.type and output_info.type.tensor_type and output_info.type.tensor_type.shape.dim:
            output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        else:
            raise ValueError("Output tensor information is incomplete or missing.")

        if hidden_info and hidden_info.type and hidden_info.type.tensor_type and hidden_info.type.tensor_type.shape.dim:
            hidden_state_shape = [dim.dim_value for dim in hidden_info.type.tensor_type.shape.dim]
        else:
            raise ValueError("Hidden state tensor information is incomplete or missing.")
    else:
        raise ValueError("Graph proto output does not have the expected number of elements.")

    # Infer batch_first by comparing output and hidden state shapes
    if output_shape[0] == hidden_state_shape[1]:  # Compare the batch dimension
        batch_first = True
    elif output_shape[1] == hidden_state_shape[1]:  # Check for sequence-first
        batch_first = False
    else:
        raise ValueError("Cannot determine batch_first from output and hidden state shapes")

    # Infer num_layers from hidden state shape
    num_layers_inferred = hidden_state_shape[0] // (2 if is_bidirectional else 1)
    num_layers = node.attributes.get('num_layers', num_layers_inferred)

    print(f"Inferred attributes for {rnn_type}:")
    print(f"Input Size: {input_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Layers: {num_layers} (inferred {num_layers_inferred})")
    print(f"Bidirectional: {is_bidirectional}")
    print(f"Dropout: {dropout}")
    print(f"Batch First: {batch_first}")

    try:
        rnn_class = _RNN_CLASS_FROM_TYPE[rnn_type]
    except KeyError as exc:
        raise NotImplementedError(f'RNN type {rnn_type} is not implemented') from exc

    torch_module = rnn_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers_inferred,
        bias=True,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=is_bidirectional,
    )
    torch_module = RNNWrapper(
        rnn_module=torch_module,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers_inferred,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=is_bidirectional,
    )

    with torch.no_grad():
        torch_module.weight_ih_l0 = weights_ih
        torch_module.weight_hh_l0 = weights_hh
        if bias_ih is not None:
            torch_module.bias_ih_l0 = bias_ih
        if bias_hh is not None:
            torch_module.bias_hh_l0 = bias_hh
    
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=node.input_values,
            outputs=node.output_values,
        ),
    )