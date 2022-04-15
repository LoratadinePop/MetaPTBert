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
from simcse.hypernet import MetaPrefixEncoderHyperNet
from simcse.prefix_encoder import PrefixEncoder, get_prefix, MetaPrefixEncoder

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
    past_key_values=None
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # transform the batch_size, 2, len to batch_size * num_sent, len, which is the format encoder can process
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len) [sent0.0, sent0.1, sent1.0, sent1.1 ...]
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Done: vanilla prefix-tuning
    prefix_attention_mask = None
    device = torch.device(input_ids.device)
    bs = input_ids.shape[0]

    if cls.use_prefix:
        past_key_values = get_prefix(cls, batch_size=bs, device=device, dropout_prob=cls.config.hidden_dropout_prob)
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    # Done: meta prefix tuning
    if cls.meta_prefix:
        past_key_values = cls.meta_prefix_encoder(
            input_ids=input_ids,
            cls=cls,
            attention_mask=attention_mask,
            encoder=encoder,
            device=device)
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # print("model input attention_mask", prefix_attention_mask.shape)
        # print("model input past_key_values", past_key_values[0].shape)

    # SimCSE's original implementations, keep unchanged.
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=prefix_attention_mask if prefix_attention_mask is not None else attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        past_key_values=past_key_values,
        model_call_this_method=cls,
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

    # Done: vanilla prefix-tuning
    prefix_attention_mask = None
    device = torch.device(input_ids.device)
    if cls.use_prefix:
        bs = input_ids.shape[0]
        past_key_values = get_prefix(cls, batch_size=bs, device=device, dropout_prob=cls.config.hidden_dropout_prob)
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    # Done: meta prefix tuning
    elif cls.meta_prefix:
        past_key_values = cls.meta_prefix_encoder(
            input_ids=input_ids,
            cls=cls,
            attention_mask=attention_mask,
            encoder=encoder,
            device=device)
        bs = input_ids.shape[0]
        prefix_attention_mask = torch.ones(bs, cls.model_args.pre_seq_len).to(device)
        prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    # Done: Hyper prefix tuning, just pass cls to the encoder() is enough
    
    # Keep original as SimCSE
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
        past_key_values=past_key_values,
        model_call_this_method=cls,
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

    def __init__(self, config, **model_kargs):
        super().__init__(config)
        self.config = config
        self.model_args = model_kargs["model_args"]
        self.use_prefix = self.model_args.prefix
        self.meta_prefix = self.model_args.meta_prefix
        self.hyper_prefix = self.model_args.hyper_prefix
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(self.config)
            print("do mlm, initialize lm_head. ", self.lm_head)


        # @1 vanilla prefix tuning
        if self.use_prefix:
            self.prefix_tokens = torch.arange(self.model_args.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(self.config, model_args=self.model_args)

        # @2 general meta prefix, conditioned on layer embedding and avg input
        if self.meta_prefix:
            self.meta_prefix_encoder = MetaPrefixEncoder(self.config, model_args=self.model_args)

        """
            @3 HyperMetaPT
            If layer_wise, will generate the weights for the MetaPrefixEncoder given a set of learnable
            layer embedding. The input to the MetaPrefixEncoder is the avg. hidden state of BERT's each layer.
            Other wise, will generate a list of (equal to BERT's layer nums) MetaPrefixEncoder for BERT's
            each layer. The input is also the avg. hidden state of BERT's each layer. This is the ablation
            experiment to not  using HyperNetwork.
        """
        if self.hyper_prefix:
            if self.model_args.layer_wise:
                self.layer_embeddings = nn.Embedding(self.config.num_hidden_layers, self.model_args.layer_embed_size)
                self.hyper_prefix_encoder = MetaPrefixEncoderHyperNet(
                                            input_dim=self.config.hidden_size, 
                                            hidden_dim=self.model_args.prefix_hidden_size, 
                                            output_dim=self.config.hidden_size * self.model_args.pre_seq_len * 2, # pre_seq_len * ( key + value )
                                            embedding_dim=self.model_args.layer_embed_size)
            else:
                # a unique encoder for each layer
                self.layer_prefix_encoder = [
                    nn.Sequential(
                        nn.Linear(self.config.hidden_size, self.model_args.prefix_hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.model_args.prefix_hidden_size, self.config.hidden_size * self.model_args.pre_seq_len * 2)
                    )
                    for _ in range(self.config.num_hidden_layers)
                ]
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