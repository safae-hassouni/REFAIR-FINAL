
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import GCN2Conv  
from torch_geometric.nn import APPNP
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TransformerConv

class GraphTransNet(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, num_layers=2, dropout=0.5, heads=4):
        super(GraphTransNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # 1ère couche Transformer
        self.convs.append(TransformerConv(in_feats, hidden_dim // heads, heads=heads, dropout=dropout))
        # couches cachées
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        # dernière couche (sortie)
        self.convs.append(TransformerConv(hidden_dim, out_feats, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # PAS de log_softmax ici, on renvoie des logits bruts
        return x

class GraphAIRNet(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout=0.5, num_layers=2, heads=1):
        super(GraphAIRNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.residuals = nn.ModuleList()

        # First layer
        self.attentions.append(GATConv(in_feats, hidden_dim, heads=heads, dropout=dropout))
        self.residuals.append(nn.Linear(in_feats, hidden_dim * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.attentions.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.residuals.append(nn.Linear(hidden_dim * heads, hidden_dim * heads))

        # Output layer
        self.attentions.append(GATConv(hidden_dim * heads, out_feats, heads=1, concat=False, dropout=dropout))
        self.residuals.append(nn.Linear(hidden_dim * heads, out_feats))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            h = self.attentions[i](x, edge_index)
            res = self.residuals[i](x)
            x = F.elu(h + res)
            if i != self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x 

class SpectralAttentionLayer(nn.Module):
    def __init__(self, channels, dropout=0.5):
        super(SpectralAttentionLayer, self).__init__()
        self.proj = nn.Linear(channels, channels)
        self.attn = nn.Linear(channels, 1)
        self.dropout = dropout

    def forward(self, x):
        h = torch.tanh(self.proj(x))
        attn_weights = F.softmax(self.attn(h), dim=0)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        return x * attn_weights

class SpecFormerNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(SpecFormerNet, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.attn_layers = nn.ModuleList([SpectralAttentionLayer(hidden_channels, dropout) for _ in range(num_layers)])
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, attn in zip(self.convs, self.attn_layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = attn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    


class PolynormerNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3, dropout=0.5):
        super(PolynormerNet, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.gcn_layers = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(K)])
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.gcn_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    

class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1, dropout=0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.appnp = APPNP(K=K, alpha=alpha)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.appnp(x, edge_index)
        return F.log_softmax(x, dim=1)

    
class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(GraphormerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        # Attention
        residual = x
        if attn_bias is not None:
            attn_output, _ = self.attn(x, x, x, attn_mask=attn_bias)
        else:
            attn_output, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_output)
        x = self.norm1(x)

        # Feedforward
        residual = x
        x = self.linear2(F.relu(self.linear1(x)))
        x = residual + self.dropout(x)
        x = self.norm2(x)

        return x


class Graphormer(nn.Module):
    def __init__(self, input_dim=3,num_layers=3, hidden_dim=128, n_heads=8, dropout=0.1, num_classes=2):
        super(Graphormer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
 
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_features, attn_bias=None):
        x = self.embedding(node_features)
        print(x.size())
        for layer in self.layers:
            x = layer(x, attn_bias)
        print(x.size())
        x = x.mean(dim=1)  # graph-level representation
        out = self.readout(x)
        return out
    

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x: [N, D]
        Q = self.query(x)  # [N, D]
        K = self.key(x)    # [N, D]
        V = self.value(x)  # [N, D]

        # Attention scores: [N, N]
        attention = torch.matmul(Q, K.transpose(0, 1)) / self.scale
        attention = F.softmax(attention, dim=-1)

        # Output: [N, D]
        out = torch.matmul(attention, V)
        return out

class SAN(nn.Module):
    def __init__(self, input_dim, embed_dim, num_classes, dropout=0.5):
        super(SAN, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.self_attention = SelfAttention(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)         # [N, embed_dim]
        x = self.self_attention(x)    # [N, embed_dim]
        x = self.dropout(x)
        x = self.classifier(x)        # [N, num_classes]
        return x
    

    
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)



def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
class GAT(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout, heads=1, args=None):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden, heads=self.heads, dropout=dropout)
        self.conv2 = GATConv(hidden * self.heads, out_channels, dropout=dropout)

    def forward(self, x, edge_index, return_attn=False):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_alpha = self.conv1(x, edge_index,return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        if return_attn:
            return x , edge_alpha[1]
        return logits


## model GCN
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, nlayer=2, args=None):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.nlayer = nlayer
        self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(in_channels, hidden_channels))
        for layer in range(self.nlayer-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha=0.1, theta=0.5, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta

        # couche d'entrée (projection linéaire)
        self.linear_in = nn.Linear(in_channels, hidden_channels)

        # couches GCNII (GCN2Conv)
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1))

        # couche de sortie
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.linear_in(x)
        x0 = x  # mettre x0 après projection
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = conv(x, x0, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear_out(x)
        return x


class FairGT(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        self.seq_len = net_params['hops']+1
        self.pe_dim = net_params['pe_dim']
        self.input_dim = net_params['in_dim']
        self.hidden_dim = net_params['hidden_dim']
        # self.ffn_dim = 
        self.ffn_dim = 2 * self.hidden_dim
        self.num_heads = net_params['n_heads']
        
        self.n_layers = net_params['n_layers']
        self.n_class = net_params['nclass']

        self.dropout_rate = net_params['dropout']
        self.attention_dropout_rate = net_params['dropout']
        
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(self.hidden_dim)

   

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        self.Linear1 = nn.Linear(int(self.hidden_dim/2), self.n_class)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)


        self.apply(lambda module: init_params(module, n_layers=self.n_layers))

    def forward(self, batched_data):


        tensor = self.att_embeddings_nope(batched_data)

        
        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        
        output = self.final_ln(tensor)
   
        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))

        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_tensor = neighbor_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()


        output = self.Linear1(torch.relu(self.out_proj(output)))

        return output
        # return torch.log_softmax(output, dim=1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None: # learned adjacency
            x = x + attn_bias

        x = torch.softmax(x, dim=3)  # learned adjacency normalized
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn] # convolution

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):

        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        
        return x

