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
    def __init__(self, rnn_module, batch_first):
        super(RNNWrapper, self).__init__()
        self.rnn_module = rnn_module
        self.batch_first = batch_first

    def forward(self, input_1, *args):
        print("Input dim: ", input_1.dim())
        if self.batch_first:
            input_1 = input_1.permute(1,0,2)

        if len(args) == 2:
            hx = args
            print("Hidden state dims: ", [h.dim() for h in hx])
            output, hidden = self.rnn_module(input_1, hx)
        else:
            output, hidden = self.rnn_module(input_1)
        print("Output dim: ", output.dim())
        if self.batch_first and output.dim() == 3:
            output = output.permute(1,0,2)
        return output, hidden

@add_converter(operation_type='RNN', version=1)
@add_converter(operation_type='RNN', version=14)
@add_converter(operation_type='LSTM', version=14)
@add_converter(operation_type='GRU', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    rnn_type = node.operation_type
    weights_ih_name = node.input_values[1]
    weights_hh_name = node.input_values[2]
    #print("Nodes: ", node.input_values)
    #print("Shape: ", [n for n in node.input_values])
    #breakpoint()
    '''
    step 1 verify the inputs and input shapes
    step 2 which of the inputs need to be passed & in what order to torch code to make it run
    
    '''
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

    # Fetch model attributes
    node_attributes = node.attributes
    input_size = node_attributes.get('input_size')
    hidden_size = node_attributes.get('hidden_size')
    num_layers = node_attributes.get('num_layers', 1)
    bidirectional = node_attributes.get('direction', 'forward') == 'bidirectional'
    dropout = node_attributes.get('dropout', 0.0)
    batch_first = node_attributes.get('batch_first', False)

    # If input_size is not set, derive it from the input tensor shape
    if input_size is None:
        input_name = node.input_values[0]
        input_tensor = graph.value_info[input_name]
        input_size = input_tensor.type.tensor_type.shape.dim[-1].dim_value

    print(f"Attributes of {rnn_type} RNN:")
    print(f"Input Size: {input_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Layers: {num_layers}")
    print(f"Bidirectional: {bidirectional}")
    print(f"Dropout: {dropout}")

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
        bidirectional=bidirectional,
    )
    torch_module = RNNWrapper(torch_module, batch_first)
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