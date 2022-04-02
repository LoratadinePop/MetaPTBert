from cProfile import label
from curses import meta
from turtle import forward
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

'''
BertConfig {
  "_name_or_path": "bert-base-uncased",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.2.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}

ModelArguments(model_name_or_path='bert-base-uncased', 
model_type=None, config_name=None, 
tokenizer_name=None, cache_dir=None, use_fast_tokenizer=True, model_revision='main',
 use_auth_token=False, prefix=False, hyper_prefix=False, pre_seq_len=4, prefix_projection=False, 
 prefix_hidden_size=512, temp=0.05, pooler_type='cls',
  hard_negative_weight=0, do_mlm=False, mlm_weight=0.1, mlp_only_train=True)

'''
class MetaPrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the instance-wise and layer-wise meta-prefix.
    '''

    def __init__(self, config, **model_kwargs) -> None:
        super().__init__()
        self.model_args = model_kwargs['model_args']
        self.config = config
        # Layer embeddings
        if self.model_args.layer_wise:
            self.layer_embedding = torch.nn.Embedding(config.num_hidden_layers, self.model_args.layer_embed_size)
        
        # meta embedding encoder
        self.meta_embed_net = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size + self.model_args.layer_embed_size 
                if self.model_args.layer_wise else config.hidden_size, self.model_args.meta_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.model_args.meta_hidden_size,  self.model_args.meta_embed_size)
        )
        # meta prefix encoder, input: meta embedding
        self.meta_net = torch.nn.Sequential(
            torch.nn.Linear(self.model_args.meta_embed_size, self.model_args.prefix_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.model_args.prefix_hidden_size, 2 * config.hidden_size * self.model_args.pre_seq_len 
            if self.model_args.layer_wise else 2 * config.hidden_size * config.num_hidden_layers * self.model_args.pre_seq_len)
            # 如果是只以avg作为meta embedding，那么需要投射到每一层（同样的输入，同样的输出），而如果是layer_wise，每一层的input都不同，因此只需要投射到该层的embedding length即可。
        )
    
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        cls = kwargs["cls"]
        batch_size = input_ids.shape[0]
        attention_mask = kwargs["attention_mask"]
        encoder = kwargs["encoder"]
        device = kwargs['device']
        embedding_layer = encoder.get_input_embeddings() # model's embedding layer

        input_embedding = embedding_layer(input_ids)
        avg_input_embedding = (input_embedding * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        past_key_values = None

        if self.model_args.layer_wise:
            # Todo: concat instance input, batch it and process it
            layer_embed = self.layer_embedding(torch.arange(self.config.num_hidden_layers).long().to(device))
            # [12, 512] -> [1, 12, 512] -> [64, 12, 512]
            layer_embed = layer_embed.unsqueeze(0).expand([batch_size, layer_embed.shape[0], layer_embed.shape[1]])
            # [64, 768] -> [64,1,768] -> [64, 12, 768]
            avg_input_embedding = avg_input_embedding.unsqueeze(1).expand([avg_input_embedding.shape[0], self.config.num_hidden_layers, avg_input_embedding.shape[-1]])
            meta_embedding = torch.cat((layer_embed, avg_input_embedding), dim=2)
            meta_embedding = self.meta_embed_net(meta_embedding)
            meta_prefix = self.meta_net(meta_embedding) # [64, 12, ?]
            past_key_values = meta_prefix.view([batch_size,-1]) # [64, ?]
            # past_key_values = meta_prefix.view(
            #     batch_size,
            #     self.config.num_hidden_layers*2,
            #     self.config.num_attention_heads,
            #     self.config.hidden_size // self.config.num_attention_heads
            # )
            # past_key_values = cls.dropout(past_key_values)
            # past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            # Todo: directly map avg. instance embedding to meta prefix
            meta_embedding = self.meta_embed_net(avg_input_embedding) # [64, 512]
            past_key_values = self.meta_net(meta_embedding)

        past_key_values = past_key_values.view(
            batch_size,
            self.model_args.pre_seq_len,
            self.config.num_hidden_layers * 2, 
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        past_key_values = cls.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

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


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        # print("last_hidden",last_hidden.shape) # 128, 32, 768
        # print("atten", attention_mask.shape) # 128, 32
        # print("atten.unsquee", attention_mask.unsqueeze(-1).shape) # 128, 32, 1
        # print("last_hidden * atten.uns", (last_hidden * attention_mask.unsqueeze(-1)).shape)
        # print("shang.sum(1)", (last_hidden * attention_mask.unsqueeze(-1)).sum(1).shape)
        # print("atten_sum-1", attention_mask.sum(-1).shape) # 128

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            # Note that we remove the mlp of original BERT
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def get_prefix(cls, batch_size, device=None):
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
    past_key_values = cls.dropout(past_key_values)
    # print(past_key_values.shape) 128, 4, 24, 12, 64
    # permute 24, 128, 12, 4, 64
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    # print(len(past_key_values)) 12
    # print(past_key_values[0].shape) 2, 128, 12, 4, 64
    return past_key_values

def cl_init(cls, config):
    # cls means self
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    past_key_values=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # TAG: transform the batch_size, 2, len to batch_size * num_sent, len, which is the format encoder can process
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len) [sent0.0, sent0.1, sent1.0, sent1.1 ...]
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Done: Apply prefix-tuning here!
    # Keep the code independent for debug
    prefix_attention_mask = None
    device = torch.device(input_ids.device)
    if cls.use_prefix:
        bs = input_ids.shape[0]
        past_key_values = get_prefix(cls, batch_size=bs, device=device)
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    # Todo: MetaPrefix
    if cls.meta_prefix:
        past_key_values = cls.meta_prefix_encoder(
            input_ids=input_ids,
            cls=cls,
            attention_mask=attention_mask,
            encoder=encoder,
            device=device)

        bs = input_ids.shape[0]
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=prefix_attention_mask if prefix_attention_mask is not None else attention_mask, # 
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        past_key_values=past_key_values
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    # Todo: How does it work?
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    labels = torch.arange(cos_sim.size(0)).long().to(torch.device(str(input_ids.device)))

    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    # print("cos_sim", cos_sim.device)
    # print("labels", labels.device)
    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    past_key_values=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # Done: Apply prefix-tuning here!
    prefix_attention_mask = None
    device = torch.device(input_ids.device)
    if cls.use_prefix:
        bs = input_ids.shape[0]
        past_key_values = get_prefix(cls, batch_size=bs, device=device)
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    # Todo: MetaPrefix
    if cls.meta_prefix:
        past_key_values = cls.meta_prefix_encoder(
            input_ids=input_ids,
            cls=cls,
            attention_mask=attention_mask,
            encoder=encoder,
            device=device)

        bs = input_ids.shape[0]
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    outputs = encoder(
        input_ids,
        attention_mask=prefix_attention_mask if prefix_attention_mask is not None else attention_mask, #
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        past_key_values=past_key_values
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        # 是否在inference的时候也加MLP
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

class PrefixBertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.config = config
        self.model_args = model_kargs["model_args"]
        self.use_prefix = self.model_args.prefix
        self.meta_prefix = self.model_args.meta_prefix
        self.bert = BertModel(config, add_pooling_layer=False)
        # print("model architecture", self.bert)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(self.config)
            print(self.lm_head)

        if self.use_prefix:
            self.prefix_tokens = torch.arange(self.model_args.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(self.config, model_args=self.model_args)
        
        if self.meta_prefix:
            self.meta_prefix_encoder = MetaPrefixEncoder(self.config, model_args=self.model_args)

        cl_init(self, self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            # input: batch_size, seq_length
            # print("eval: ", attention_mask.shape, input_ids.shape)
            # batch_size = input_ids.shape[0]
            # past_key_values = self.get_prefix(batch_size=batch_size, device=input_ids.device)
            # prefix_attention_mask = torch.ones(batch_size, self.model_args.pre_seq_len).to(torch.device(input_ids.device))
            # # print("attention_mask ", attention_mask.shape) # batch_size, 2, length
            # # print("prefix_attention_mask ", prefix_attention_mask.shape) # batch_size, prefix_length
            # prefix_attention_mask = prefix_attention_mask.unsqueeze(1).expand([prefix_attention_mask.shape[0],1,prefix_attention_mask.shape[1]])
            # # print("prefix_attention_mask ",prefix_attention_mask.shape) # batch_size, 2, prefix_length
            # attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=2)
            return sentemb_forward(
                self, 
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # # input: batch_size, 2(positive pair), seq_length
            # print("train: ", attention_mask.shape, input_ids.shape)
            # batch_size = input_ids.shape[0]
            # past_key_values = self.get_prefix(batch_size=batch_size*2, device=input_ids.device)
            # prefix_attention_mask = torch.ones(batch_size, self.model_args.pre_seq_len).to(torch.device(input_ids.device))
            # # print("attention_mask ", attention_mask.shape) # batch_size, 2, length
            # # print("prefix_attention_mask ", prefix_attention_mask.shape) # batch_size, prefix_length
            # prefix_attention_mask = prefix_attention_mask.unsqueeze(1).expand([prefix_attention_mask.shape[0],2,prefix_attention_mask.shape[1]])
            # # print("prefix_attention_mask ",prefix_attention_mask.shape) # batch_size, 2, prefix_length
            # attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=2)
            return cl_forward(
                self, 
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )