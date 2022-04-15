import torch
import torch.nn.functional as F

embedding_dim = 32
input_dim = 768
output_dim = 768 * 2 * 8
hidden_dim = 512
batch_size = 256

from simcse.hypernet import MetaPrefixEncoderHyperNet

device = torch.device("cuda:0")

embedding = torch.nn.Embedding(12, embedding_dim).to(device)

hypernet = MetaPrefixEncoderHyperNet(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    embedding_dim=embedding_dim
).to(device)

loss = torch.nn.CrossEntropyLoss()

for i in range(100):
    i %= 12
    layer_embedding = embedding(torch.LongTensor([i]).to(device))
    input = torch.rand(batch_size, input_dim, requires_grad=True).to(device)
    hyperweight = hypernet(layer_embedding)
    hidden_state = F.linear(
        input,
        hyperweight.down_projection.weight,
        bias=hyperweight.down_projection.bias)
    # print(hidden_state.shape)
    output = F.linear(
        hidden_state,
        hyperweight.up_projection.weight,
        bias=hyperweight.up_projection.bias)
    print(output.shape)

    target = torch.empty(batch_size, dtype=torch.long).random_(output_dim).to(device)
    a = loss(output, target)
    a.backward()
