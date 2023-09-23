import random

import torch
import torch.nn as nn
from transformers import T5EncoderModel, RobertaModel

from adapterblock import Adapter


class PLMAdapter(nn.Module):
    def __init__(self, args, vocab_size):
        super(PLMAdapter, self).__init__()
        self.args = args
        self.adapter_position_embed = nn.Embedding(args.max_len, args.hidden_size)

        adapter = None
        adapter_list = list(range(args.num_encoder_layer))
        if args.add_adapter:
            adapter = self.get_adapter(args)
        adapter_parameter = {'add_adapter': args.add_adapter, 'adapter_list': adapter_list,
                             'adapter_modules': adapter}

        if args.model_type == 't5':
            self.plm_encoder = T5EncoderModel.from_pretrained(args.t5_model_type, adapter_parameter)

        elif args.model_type == 'roberta':
            self.plm_encoder = RobertaModel.from_pretrained(args.roberta_model_type,
                                                            adapter_parameter=adapter_parameter)
        else:
            raise TypeError

        self.scorer1 = nn.Linear(args.hidden_size, 128)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.scorer2 = nn.Linear(128, 1)

        self.mask_proj = MaskHead(args.hidden_size, vocab_size)
        self.relation_proj = nn.Linear(args.hidden_size * 2, args.hidden_size)

    def get_input_embeddings(self):
        embed = self.plm_encoder.get_input_embeddings()
        return embed

    def create_relation_embed(self, relation2id, tokenizer, plm_embed):
        rel_embeds = []
        for rel in relation2id.keys():
            token = tokenizer(rel, return_tensors='pt').input_ids.to(self.args.local_rank)
            embed = plm_embed(token).mean(1).squeeze()
            rel_embeds.append(embed)

        return torch.stack(rel_embeds)

    def get_adapter(self, args):
        adapter = Adapter(args, args.weight_init_option, args.adapter_scalar, args.adapter_dropout,
                          args.down_size, args.adapter_layernorm_option)

        return adapter

    def embed_knowledge(self, knowledge_id: torch.Tensor, plm_embed):
        '''
        Args:
            knowledge_id: [batch_size, num_knowledge, max_len]

        Returns:
            knowledge_embedding: [batch_size, num_knowledge, max_len, hidden_size]
        '''
        knowledge_embedding = plm_embed(knowledge_id)

        return knowledge_embedding

    def mask_head(self, adapter_output):
        return self.mask_proj(adapter_output)

    def relation_head(self, adapter_output, head_tail_index, rel_ids, sign, k):
        '''
        Args:
            adapter_output: [batch_size, num_knowledge, max_len, hidden_size]
            head_tail_index: [batch_size, num_knowledge, 2, 2]
            rel_ids: [batch_size, num_knowledge]
            sign: [batch_size]
        Returns:
            [(relation_tensor1, rel_id1), (relation_tensor2, rel_id2), ...]
        '''
        result_for_contrastive = []
        for b in range(sign.shape[0]):
            if sign[b].item() != 0:
                if rel_ids[b, k] < 0:
                    continue
                else:
                    head_index, tail_index = head_tail_index[b, k, 0, :], head_tail_index[b, k, 1, :]
                    head_tensor = adapter_output[b, head_index[0]:head_index[1], :].mean(0)
                    tail_tensor = adapter_output[b, tail_index[0]:tail_index[1], :].mean(0)

                    relation_tensor = self.relation_proj(torch.concat((head_tensor, tail_tensor), dim=-1))
                    rel_id = rel_ids[b, k].item()

                    result_for_contrastive.append((relation_tensor, rel_id))

            else:
                continue

        return result_for_contrastive

    def forward(self, token_ids, attention_masks, knowledge_id, knowledge_attn_mask, sign, head_tail_index, rel_ids,
                plm_embed):
        '''
        Args:
            token_ids: [batch_size, num_class, max_len]
            attention_masks: [batch_size, num_class, max_len]
            knowledge_id: [batch_size, num_knowledge, max_len]
            knowledge_attn_mask: [batch_size, num_knowledge, max_len]
            rel_ids: [batch_size, num_knowledge]
        Returns:

        '''
        adapter_inputs, adapter_mask = None, None
        if self.args.add_adapter:
            adapter_inputs = self.embed_knowledge(knowledge_id, plm_embed)
            adapter_mask = knowledge_attn_mask
            pos = torch.arange(knowledge_id.shape[2], dtype=torch.long, device=knowledge_id.device)
            pos = pos.unsqueeze(0).expand((knowledge_id.shape[0], knowledge_id.shape[-1]))
            pos_enc = self.adapter_position_embed(pos)

        num_class = token_ids.size(1)

        plm_logits_outs, ada_outs = [], []
        for idx in range(num_class):
            if self.args.add_adapter:
                id_k = random.choice(list(range(knowledge_id.shape[1])))

                adapter_input = adapter_inputs[:, id_k, :, :]

                encoder_out, adapter_out = self.plm_encoder(input_ids=token_ids[:, idx, :],
                                                            attention_mask=attention_masks[:, idx, :],
                                                            adapter_input=adapter_input + pos_enc,
                                                            adapter_mask=adapter_mask[:, id_k, :])

                last_hidden_state = encoder_out.last_hidden_state.mean(1)

                plm_logits_outs.append(last_hidden_state)
                ada_outs.append(adapter_out)

            else:
                tmp_enc_out, tmp_adapter_out = self.t5_encoder(input_ids=token_ids[:, idx, :],
                                                               attention_mask=attention_masks[:, idx, :],
                                                               adapter_input=adapter_inputs,
                                                               adapter_mask=adapter_mask)
                last_hidden_state = tmp_enc_out.last_hidden_state.mean(1)
                plm_logits_outs.append(last_hidden_state)

        plm_logits = self.scorer1(torch.stack(plm_logits_outs, dim=1))
        plm_logits = self.act(plm_logits)

        plm_logits = self.scorer2(plm_logits)

        if self.args.add_adapter:
            ada_outs = torch.stack(ada_outs)[0]
            mask_adapter_out = self.mask_head(ada_outs)

            result_for_contrastive = self.relation_head(ada_outs, head_tail_index, rel_ids, sign, id_k)

        else:
            mask_adapter_out, result_for_contrastive, id_k = None, None, None

        return plm_logits, mask_adapter_out, result_for_contrastive, id_k


class MaskHead(nn.Module):
    '''
    modify from AlbertMLMHead
    '''

    def __init__(self, hidden_size, vocab_size):
        super(MaskHead, self).__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.GELU()
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states
