# pylint: disable=missing-docstring
import torch
from torch import nn

from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxBaseElementWise(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(self, op_type: str):
        super().__init__()
        self.op_type = op_type
        self._op_type = op_type

    @staticmethod
    def _broadcast_shape(*tensors: torch.Tensor):
        shapes = [t.shape for t in tensors]
        broadcast_shape = torch.broadcast_shapes(*shapes)
        return broadcast_shape

    def apply_reduction(self, *tensors: torch.Tensor) -> torch.Tensor:
        del tensors
        raise NotImplementedError

    def forward(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        if len(input_tensors) == 1:
            # If there is a single element, return it (no op).
            # Also, no need for manually building the ONNX node.
            return input_tensors[0]

        def _forward() -> torch.Tensor:
            return self.apply_reduction(*input_tensors)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, self._op_type, *input_tensors, {})

        return _forward()
