import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self,
                 args,
                 init_option="bert",
                 adapter_scalar="2",
                 dropout=0.0,
                 down_size=32,
                 adapter_layernorm_option='both'):
        '''
        adapter_layernorm_option: both, in, out, none
        '''
        super(Adapter, self).__init__()

        self.adapterlayer = AdapterLayer(args, init_option, adapter_scalar, dropout, down_size,
                                         adapter_layernorm_option)
        self.gate = PLMGate(args.hidden_size)

    def forward(self, adapter_hidden_states, adapter_mask, plm_hidden_states, plm_mask):
        '''

        Args:
            adapter_hidden_states: [batch_size, max_len, 768]
            adapter_mask: [batch_size, max_len]
            plm_hidden_states:
            plm_mask:
        Returns:

        '''
        plm_weight = self.gate(plm_hidden_states.mean(1))
        plm_out, adapter_out = self.adapterlayer(adapter_hidden_states, adapter_mask,
                                                 torch.mul(plm_weight.unsqueeze(-1), plm_hidden_states),
                                                 plm_mask)

        return plm_out, adapter_out


class AdapterLayer(nn.Module):
    def __init__(self,
                 args,
                 init_option="bert",
                 adapter_scalar="2",
                 dropout=0.0,
                 down_size=32,
                 adapter_layernorm_option='both'):
        super(AdapterLayer, self).__init__()
        self.args = args
        self.n_embd = args.hidden_size

        self.basic_adapter = AttentionAdapter(self.n_embd, down_size, dropout, adapter_scalar, args.nhead,
                                              args.num_layers, init_option, adapter_layernorm_option)

    def forward(self, hidden_states, adapter_mask, plm_hidden_states, plm_mask, add_residual=True):
        plm_length = plm_hidden_states.shape[1]
        concat_hidden_states = torch.cat([plm_hidden_states, hidden_states], dim=1)
        concat_mask = torch.cat([plm_mask, adapter_mask], dim=1)
        out = self.basic_adapter(concat_hidden_states, concat_mask, add_residual)

        return out[:, :plm_length, :], out[:, plm_length:, :]


class PLMGate(nn.Module):
    def __init__(self, hidden_size):
        super(PLMGate, self).__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, plm_embed):
        plm_embed_weight = F.sigmoid(self.gate(plm_embed))

        return plm_embed_weight


class AttentionAdapter(nn.Module):
    def __init__(self,
                 n_embd,
                 down_size,
                 dropout,
                 scale="2",
                 nhead=8,
                 num_layers=1,
                 init_option="bert",
                 adapter_layernorm_option='both'):
        super(AttentionAdapter, self).__init__()
        self.down_proj = nn.Linear(n_embd, down_size)

        self.non_linear_func = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=down_size, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.up_proj = nn.Linear(down_size, n_embd)

        if init_option == "bert":
            self.apply(init_bert_weights)
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

        self.dropout = dropout
        self.scale = scale
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm = nn.LayerNorm(n_embd)

    def forward(self, x, adapter_mask, add_residual=True, residual=None):
        residual = x if residual == None else residual

        if self.adapter_layernorm_option == 'both' or self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm(x)  # [both, in, out, none]

        down = self.down_proj(x.float())
        down = self.encoder(down, src_key_padding_mask=adapter_mask)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * float(self.scale)

        if self.adapter_layernorm_option == 'both' or self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
