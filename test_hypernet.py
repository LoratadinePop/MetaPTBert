import torch
import torch.nn.functional as F
from simcse.hypernet import MetaPrefixEncoderHyperNet

# device = torch.device("cuda:0")

# embedding = torch.nn.Embedding(12, 32).to(device)

# hypernet = MetaPrefixEncoderHyperNet(
#     input_dim=768,
#     hidden_dim=64,
#     output_dim=64,
#     embedding_dim=32
# ).to(device)

# emb = embedding(torch.LongTensor(torch.arange(12)).to(device))
# hypernet(emb)


F.linear(
    torch.rand([128, 768]),
    weight=torch.rand([12, 64, 768]),
    bias=torch.rand([12,64])
)