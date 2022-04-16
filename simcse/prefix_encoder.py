import torch
import torch.nn as nn


"""
    Implementation of meta prefix encoder.
    It provides two options:
        1. layer_wise
        2. NOT layer_wise
    Both of them are condition on input instance.
    If layer_wise, the MetaEmbedEncoder will concat the average input embeddings with layer_embeddings
    to generate meta_embedding, otherwise just use the average input embeddings to generate meta_embedding.
    Afterwards, the MetaPrefixEncoder would leverage the meta_embedding to generate instance-wise or 
    layer-instance-wise prefix for each layer.
"""
class MetaPrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the instance-wise and layer-wise meta-prefix.
    '''

    def __init__(self, config, **model_kwargs) -> None:
        super().__init__()
        self.model_args = model_kwargs['model_args']
        self.config = config
        # layer embeddings
        if self.model_args.layer_wise:
            # TODO: Whether to use a linear layer to parameterize the layer embeddings? No in HyperFormer
            self.layer_embedding = torch.nn.Embedding(config.num_hidden_layers, self.model_args.layer_embed_size)
        
        # meta embedding encoder
        # Todo: Whether to use a linear layer to parameterize the avg. input embeddings to stabilize the training?
        # Tag: This MetaEmbedEncoder doesn't needed to be parameterize by HyperNetworks.
        self.MetaEmbedEncoder = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size + self.model_args.layer_embed_size if self.model_args.layer_wise else config.hidden_size, self.model_args.meta_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.model_args.meta_hidden_size,  self.model_args.meta_embed_size)
        )
        # meta prefix encoder, input: meta embedding
        self.MetaPrefixEncoder = torch.nn.Sequential(
            torch.nn.Linear(self.model_args.meta_embed_size, self.model_args.prefix_hidden_size),
            torch.nn.Tanh(), #or Tanh()
            torch.nn.Linear(self.model_args.prefix_hidden_size, 2 * config.hidden_size * self.model_args.pre_seq_len 
            if self.model_args.layer_wise else 2 * config.hidden_size * config.num_hidden_layers * self.model_args.pre_seq_len)
        )
    
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        cls = kwargs["cls"]
        batch_size = input_ids.shape[0]
        attention_mask = kwargs["attention_mask"]
        encoder = kwargs["encoder"]
        device = kwargs['device']
        embedding_encoder = encoder.get_input_embeddings() # model's embedding layer

        input_embedding = embedding_encoder(input_ids)
        # bs, hidden_size
        avg_input_embedding = (input_embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        past_key_values = None
        if self.model_args.layer_wise:
            # Done: concat instance input, batch it and process it
            # num_hidden_layers, layer embedding_size
            layer_embed = self.layer_embedding(torch.arange(self.config.num_hidden_layers).long().to(device))
            # [12, 512] -> [1, 12, 512] -> [64, 12, 512]
            layer_embed = layer_embed.unsqueeze(0).expand([batch_size, layer_embed.shape[0], layer_embed.shape[1]])
            # [64, 768] -> [64,1,768] -> [64, 12, 768]
            avg_input_embedding = avg_input_embedding.unsqueeze(1).expand([avg_input_embedding.shape[0], self.config.num_hidden_layers, avg_input_embedding.shape[-1]])
            meta_embedding = torch.cat((layer_embed, avg_input_embedding), dim=2)
            meta_embedding = self.MetaEmbedEncoder(meta_embedding)
            meta_prefix = self.MetaPrefixEncoder(meta_embedding) # [64, 12, ?]
            past_key_values = meta_prefix.view([batch_size,-1]) # [64, ?] view是按顺序排列的
        else:
            # Done: directly map avg. input embedding to meta prefix
            meta_embedding = self.MetaEmbedEncoder(avg_input_embedding) # [64, 512]
            past_key_values = self.MetaPrefixEncoder(meta_embedding)

        past_key_values = past_key_values.view(
            batch_size,
            self.model_args.pre_seq_len,
            self.config.num_hidden_layers * 2, 
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        past_key_values = nn.functional.dropout(past_key_values, p=self.config.hidden_dropout_prob)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

"""
    Vanilla Prefix-Tuning.
"""
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config, **model_kargs):
        super().__init__()
        self.model_args = model_kargs["model_args"]
        self.prefix_projection = self.model_args.prefix_projection
        if self.prefix_projection:
            # Global prompt
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(self.model_args.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, self.model_args.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.model_args.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(self.model_args.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


"""
    Used cooperated with PrefixEncoder
"""
def get_prefix(cls, batch_size, device=None, dropout_prob=0):
    prefix_tokens = cls.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
    past_key_values = cls.prefix_encoder(prefix_tokens)
    # bsz, seqlen, _ = past_key_values.shape
    past_key_values = past_key_values.view(
        batch_size,
        cls.model_args.pre_seq_len,
        cls.config.num_hidden_layers * 2, 
        cls.config.num_attention_heads,
        cls.config.hidden_size // cls.config.num_attention_heads
    )
    past_key_values = nn.functional.dropout(past_key_values, p=dropout_prob)
    # print(past_key_values.shape) 128, 4, 24, 12, 64
    # permute 24, 128, 12, 4, 64
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    # print(len(past_key_values)) 12
    # print(past_key_values[0].shape) 2, 128, 12, 4, 64
    return past_key_values

