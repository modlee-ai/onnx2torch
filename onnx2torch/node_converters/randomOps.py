import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OperationConverterResult


_RANDOM_OPS = {
    'RandomNormal': torch.randn,
    'RandomNormalLike': torch.randn_like,
    'RandomUniform': torch.rand,
    'RandomUniformLike': torch.rand_like,
}

@add_converter(operation_type='RandomNormal', version=1)
@add_converter(operation_type='RandomNormalLike', version=9)
@add_converter(operation_type='RandomUniform', version=1)
@add_converter(operation_type='RandomUniformLike', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    op_type = node.operation_type
    torch_op = _RANDOM_OPS[op_type]
    if op_type.endswith('Like'):
        input_name = node.input_values[0]
        input_tensor = graph.get_input_value_by_name(input_name)
        output = torch_op(input_tensor.size(), **node.attributes)
    else:
        output = torch_op(**node.attributes)
    return OperationConverterResult(torch_module= output)