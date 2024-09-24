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
    def __init__(self, rnn_module, batch_first, bidirectional):
        super(RNNWrapper, self).__init__()
        self.rnn_module = rnn_module
        self.batch_first = batch_first
        self.bidirectional = bidirectional

    def forward(self, input_1, *args):
        if not isinstance(input_1, torch.Tensor):
            raise TypeError(f"Expected input_1 to be a tensor, but got {type(input_1)}")

        if self.batch_first:
            input_1 = input_1.permute(1, 0, 2)

        if len(args) == 2:
            hx = args
            output, hidden = self.rnn_module(input_1, hx)
        else:
            output, hidden = self.rnn_module(input_1)

        if self.batch_first and output.dim() == 3:
            output = output.permute(1, 0, 2)
        return output, hidden


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

    # Fetch attributes from the node or infer them
    input_size = node.attributes.get('input_size')
    print(f"Input Size: {input_size}")
    breakpoint()
    if input_size is None and node.input_values:
        # Infer input size from the first input tensor (only once)
        input_name = node.input_values[0]
        input_tensor = graph.value_info.get(input_name, None)
        if input_tensor:
            input_size = input_tensor.type.tensor_type.shape.dim[-1].dim_value

    hidden_size = node.attributes.get('hidden_size')
    num_layers = node.attributes.get('num_layers', 1)
    direction = node.attributes.get('direction', 'forward')
    dropout = node.attributes.get('dropout', 0.0)
    batch_first = node.attributes.get('batch_first', False)

    # Extract the hidden state shape from the output to infer additional attributes
    hidden_state_shape = [dim.dim_value for dim in graph.proto.output[1].type.tensor_type.shape.dim]
    is_bidirectional = direction == 'bidirectional'
    num_layers_inferred = hidden_state_shape[0] // (2 if is_bidirectional else 1)

    # Use inferred values if the attributes are missing
    num_layers = num_layers or num_layers_inferred

    # Infer batch_first from shape if necessary
    batch_size = hidden_state_shape[1] if len(hidden_state_shape) > 1 else None
    batch_first_inferred = hidden_state_shape[1] == batch_size if batch_size else False
    batch_first = batch_first if batch_first is not None else batch_first_inferred

    # Only print once after all attributes are inferred
    print(f"Attributes of {rnn_type} RNN:")
    print(f"Input Size: {input_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Layers: {num_layers}")
    print(f"Bidirectional: {is_bidirectional}")
    print(f"Dropout: {dropout}")
    print(f"Batch First: {batch_first}")

    if input_size is None or hidden_size is None:
        raise ValueError(f"Invalid RNN configuration: input_size and hidden_size must be defined (got input_size={input_size}, hidden_size={hidden_size})")

    try:
        rnn_class = _RNN_CLASS_FROM_TYPE[rnn_type]
    except KeyError as exc:
        raise NotImplementedError(f'RNN type {rnn_type} is not implemented') from exc

    torch_module = rnn_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=is_bidirectional,
    )
    torch_module = RNNWrapper(torch_module, batch_first, is_bidirectional)
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
