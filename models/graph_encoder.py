import sys

import torch
import torch.nn as nn
import math

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
    def forward(self, input):
        return {'data': input['data'] + self.module(input),
                'mask': input['mask'],
                'graph_size': input['graph_size'],
                'evaluate': input['evaluate']}


class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_size, feed_forward_size):
        super(FeedForwardLayer, self).__init__()
        if feed_forward_size > 0:
            self.layer = nn.Sequential(nn.Linear(embedding_size, feed_forward_size),
                                       nn.ReLU(),
                                       nn.Linear(feed_forward_size, embedding_size))
        else:
            self.layer = nn.Linear(embedding_size, embedding_size)


    def forward(self, input):
        module_input = input['data']
        module_input_size = module_input.size()
        if len(module_input_size) != 2:
            module_input = module_input.view(-1, module_input.size(-1))
        return self.layer(module_input).view(-1, *module_input_size[1:])



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_heads,
                 input_size,
                 embedding_size,
                 value_size=None,
                 key_size=None,):
        super(MultiHeadAttention, self).__init__()

        if value_size == None:
            value_size = embedding_size // n_heads
        if key_size == None:
            key_size = embedding_size // n_heads

        self.n_heads = n_heads
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.value_size = value_size
        self.key_size = key_size

        self.norm_factor = 1 / math.sqrt(key_size)

        self.W_query = nn.Linear(input_size, key_size, bias=False)
        self.W_key = nn.Linear(input_size, key_size, bias=False)
        self.W_value = nn.Linear(input_size, value_size, bias=False)
        #self.W_query = nn.Parameter(torch_util.Tensor(n_heads, input_dim, key_size))
        #self.W_key = nn.Parameter(torch_util.Tensor(n_heads, input_dim, key_size))
        #self.W_value = nn.Parameter(torch_util.Tensor(n_heads, input_dim, value_size))

        if embedding_size is not None:
            #self.W_out = nn.Parameter(torch_util.Tensor(n_heads, value_size, embedding_dim))
            self.W_out = nn.Linear(key_size, embedding_size)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, h=None):
        """
        q: query [batch_size, n_query, input_dim]
        h: key and value [batch_size, graph_size, input_dim]
        mask: [batch_size, n_query, graph_size], negative adjacency
        """
        q = input['data']
        mask = input['mask']
        graph_size = input['graph_size']
        evaluate = input['evaluate']

        if h == None:
            h = q  # compute self attention

        batch_size = int(q.size()[0] / graph_size)
        input_size = h.size()[-1]
        n_query = graph_size

        assert input_size == self.input_size

        hflat = h.contiguous().view(-1, input_size)
        qflat = q.contiguous().view(-1, input_size)

        v_shape = (self.n_heads, batch_size, graph_size, -1)
        k_shape = (self.n_heads, batch_size, graph_size, -1)
        q_shape = (self.n_heads, batch_size, n_query, -1)

        #Q = torch_util.matmul(qflat, self.W_query).view(q_shape)
        #K = torch_util.matmul(hflat, self.W_key).view(k_shape)
        #V = torch_util.matmul(hflat, self.W_value).view(v_shape)
        Q = self.W_query(qflat).view(q_shape)
        K = self.W_key(hflat).view(k_shape)
        V = self.W_value(hflat).view(v_shape)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  #[n_heads, batch_size, n_query, graph_size]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat((1, graph_size, 1)).bool()
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf

        attention = torch.softmax(compatibility, dim=-1)  #[n_heads, batch_size, n_query, graph_size]

        if mask is not None:
            attention_ = attention.clone()
            attention_[mask] = 0
            attention = attention_

        heads = torch.matmul(attention, V) #[n_heads, batch_size, n_query, input_size]

        #out = torch_util.mm(
        #    heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.value_size),
        #    self.W_out.view(-1, self.embedding_size)
        #).view(batch_size, n_query, self.embedding_size)
        out = self.W_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.value_size)
                         ).view(batch_size * n_query, self.embedding_size)

        return out



class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(self,
                 n_heads,
                 embedding_size,
                 feed_forward_hidden=128,
                 ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_size=embedding_size,
                    embedding_size=embedding_size,
                )
            ),
            SkipConnection(
                FeedForwardLayer(embedding_size, feed_forward_hidden)
            )
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(self,
                 n_heads,
                 embedding_size,
                 n_layers,
                 graph_size,
                 node_dim=None,
                 feed_forward_hidden=128,
                 ):
        
        super(GraphAttentionEncoder, self).__init__()

        self.init_embed = nn.Linear(node_dim, embedding_size) if node_dim is not None else None
        self.graph_size = graph_size
        self.layers = nn.Sequential(
            *(MultiHeadAttentionLayer(n_heads, embedding_size, feed_forward_hidden)
              for _ in range(n_layers))
        )


    def forward(self, x, mask=None, evaluate=None):
        '''
        x: [batch, graph_size, feature_size]
        return:
        node_embedding: [batch, graph_size, embedding_size]
        '''

        node_embedding = self.init_embed(x.reshape(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        data = {'data': node_embedding, 'mask': mask, 'graph_size': self.graph_size, 'evaluate': evaluate}
        node_embedding = self.layers(data)['data']

        return node_embedding, node_embedding.view(int(node_embedding.size(0)/self.graph_size), self.graph_size, -1).mean(dim=1)

