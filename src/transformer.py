import torch
import torch.nn as nn
import copy


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed):

        tgt2, global_attention_weights = self.multihead2(query=global_input+position_embed, key=global_input+position_embed,
                                                         value=global_input, key_padding_mask=input_key_padding_mask)
        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, global_input, input_key_padding_mask, position_embed):

        output = global_input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(output, input_key_padding_mask, position_embed)
            weights[i] = global_attention_weights

        if self.num_layers>0:
            return output, weights
        else:
            return output, None





class transformer(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, dec_layer_num=3, embed_dim=200, nhead=4, dim_feedforward=224,
                 dropout=0.1):
        super(transformer, self).__init__()

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)

        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        self.position_embedding = nn.Embedding(2, embed_dim) #present and next frame
        nn.init.uniform_(self.position_embedding.weight)


    def forward(self, current_h, h):

        b, f = current_h.shape 
        l = 2
        global_input = torch.zeros([l, b, f]).to(current_h.device)
        position_embed = torch.zeros([l , b , f]).to(current_h.device)

        global_input = torch.cat((current_h.unsqueeze(0),h.unsqueeze(0)), 0)
        position_embed_0  = self.position_embedding.weight[0].unsqueeze(0).repeat(b,1).unsqueeze(0)
        position_embed_1  = self.position_embedding.weight[1].unsqueeze(0).repeat(b,1).unsqueeze(0)
        position_embed = torch.cat((position_embed_0,position_embed_1), 0)
        global_masks = torch.zeros(b,l).to(current_h.device)
        # temporal decoder
        global_output, global_attention_weights = self.global_attention(global_input, global_masks>0, position_embed)

        output = global_output[1]
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
