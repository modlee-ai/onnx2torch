import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx2torch import convert


# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.pos_encoder = nn.TransformerEncoderLayer(model_dim, num_heads, dim_feedforward=512, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        )
        self.fc_out = nn.Linear(model_dim, output_dim)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(
            src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = self.fc_out(output)
        return output

# Hyperparameters
input_dim = 1000  # Vocabulary size
model_dim = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
output_dim = 1000  # Vocabulary size
dropout = 0.1

# Initialize the model
model = TransformerModel(input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim, dropout)
model.eval()

# Create dummy input data
src = torch.randint(0, input_dim, (10, 32))  # (sequence_length, batch_size)
tgt = torch.randint(0, input_dim, (20, 32))  # (sequence_length, batch_size)

# Export the model to ONNX
onnx_model_path = "transformer_model.onnx"
torch.onnx.export(
    model,
    (src, tgt),
    onnx_model_path,
    input_names=["src", "tgt"],
    output_names=["output"],
    dynamic_axes={
        "src": {0: "sequence_length", 1: "batch_size"},
        "tgt": {0: "sequence_length", 1: "batch_size"},
        "output": {0: "sequence_length", 1: "batch_size"}
    },
    opset_version=14
)

print(f"Model has been converted to ONNX format and saved as '{onnx_model_path}'")

model = onnx.load(onnx_model_path)
onnx.checker.check_model(model)
breakpoint()