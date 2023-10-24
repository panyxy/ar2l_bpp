import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import NamedTuple

from .graph_encoder import GraphAttentionEncoder
from .position_encoder import PositionalEncoder
from distributions import FixedCategorical, FixedNormal, SquashedNormal
from torch.distributions import TransformedDistribution, SigmoidTransform, AffineTransform
from utils import DecodeObs4Place, DecodeObs4Adv, DecodeObs4Critic

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AttentionModelFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],
                glimpse_val=self.glimpse_val[:, key],
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)

class AttentionModel(nn.Module):

    def __init__(self,
                 obs_shape,
                 action_space,
                 embedding_size,
                 hidden_size,
                 n_encode_layers,
                 n_heads = 1,
                 bin_node_num=None,
                 next_item_num=1,
                 candidate_pos_num=None,
                 node_dim=6,

                 tanh_clipping = 10,
                 mask_inner=False,
                 mask_logits=False,

                 model_ctg = None,
                 policy_ctg = None,
                ):

        super(AttentionModel, self).__init__()

        self.obs_space = obs_shape
        self.action_space = action_space
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_encode_layers = n_encode_layers
        self.n_heads = n_heads

        self.bin_node_num = bin_node_num
        self.next_item_num = next_item_num
        self.candidata_pos_num = candidate_pos_num
        self.node_dim = node_dim

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.decode_type = None
        self.temp = 1.
        self.model_ctg = model_ctg
        self.policy_ctg = policy_ctg

        graph_size = self.bin_node_num + self.next_item_num + self.candidata_pos_num
        assert graph_size * self.node_dim == obs_shape.shape[0]

        activate, act_name = nn.LeakyReLU, 'leaky_relu'
        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain(act_name))

        if self.next_item_num > 1 and self.policy_ctg == 'place':
            self.position_embedder = PositionalEncoder(
                self.next_item_num,
                self.embedding_size,
            )
        else:
            self.position_embedder = None

        if model_ctg == 'actor':
            self.init_bin_node_embed = nn.Sequential(
                init_(nn.Linear(self.node_dim-2, 32)),
                activate(),
                init_(nn.Linear(32, embedding_size))
            )

            self.init_next_item_embed = nn.Sequential(
                init_(nn.Linear(self.node_dim-5, 32)),
                activate(),
                init_(nn.Linear(32, embedding_size))
            )

            if self.candidata_pos_num:
                self.init_candidate_pos_embed = nn.Sequential(
                    init_(nn.Linear(self.node_dim - 1, 32)),
                    activate(),
                    init_(nn.Linear(32, embedding_size))
                )

            self.graph_embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embedding_size=embedding_size,
                n_layers=self.n_encode_layers,
                graph_size=graph_size,
            )

            self.project_node_embeddings = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
            self.project_fixed_context = nn.Linear(embedding_size, embedding_size, bias=False)

        elif model_ctg == 'critic':
            self.init_bin_node_embed = nn.Sequential(
                init_(nn.Linear(self.node_dim-2, 32)),
                activate(),
                init_(nn.Linear(32, embedding_size))
            )

            self.init_next_item_embed = nn.Sequential(
                init_(nn.Linear(self.node_dim-5, 32)),
                activate(),
                init_(nn.Linear(32, embedding_size))
            )

            self.graph_embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embedding_size=embedding_size,
                n_layers=self.n_encode_layers,
                graph_size=self.bin_node_num + self.next_item_num,
            )

            self.project_state_context = nn.Linear(embedding_size, embedding_size, bias=False)

        else:
            raise 'Invalid Model Category'



        assert embedding_size % n_heads == 0

    def actor_place_forward(self, input, deterministic=False, evaluate_action=False, normFactor=1, evaluate=False,):

        bin_node, next_item, candidate_pos, valid_mask, full_mask = \
            DecodeObs4Place(input, self.bin_node_num, self.next_item_num, self.candidata_pos_num, self.node_dim, )

        invalid_mask = 1 - valid_mask
        valid_length = full_mask.sum(1)
        full_mask = 1 - full_mask

        batch_size = input.size(0)
        graph_size = input.size(1)
        bin_node_size = bin_node.size(1)
        next_item_size = next_item.size(1)
        candidate_pos_size = candidate_pos.size(1)
        assert graph_size == bin_node_size + next_item_size + candidate_pos_size

        bin_node = bin_node.contiguous().view(batch_size * bin_node_size, self.node_dim - 2) * normFactor
        next_item = next_item.contiguous().view(batch_size * next_item_size, self.node_dim - 5) * normFactor
        candidate_pos = candidate_pos.contiguous().view(batch_size * candidate_pos_size, self.node_dim - 1) * normFactor

        bin_node_embedding = self.init_bin_node_embed(bin_node).reshape((batch_size, -1, self.embedding_size))
        next_item_embedding = self.init_next_item_embed(next_item).reshape((batch_size, -1, self.embedding_size))
        next_item_embedding = self.position_embedder(next_item_embedding) if self.position_embedder else next_item_embedding

        candidate_pos_embedding = self.init_candidate_pos_embed(candidate_pos).reshape(
            (batch_size, -1, self.embedding_size))

        init_embedding = torch.cat((bin_node_embedding, next_item_embedding, candidate_pos_embedding), dim=1).view(
            batch_size * graph_size, self.embedding_size)

        embeddings, _ = self.graph_embedder(init_embedding, mask=full_mask, evaluate=evaluate)
        embedding_shape = (batch_size, graph_size, embeddings.size(-1))

        action_log_probs, actions, entropy, dist, hidden = \
            self._inner(embeddings,
                        deterministic=deterministic,
                        evaluate_action=evaluate_action,
                        shape=embedding_shape,
                        invalid_mask=invalid_mask,
                        full_mask=full_mask,
                        valid_length=valid_length,
                        valid_logit_range=(self.bin_node_num+self.next_item_num, self.bin_node_num+self.next_item_num+self.candidata_pos_num),
                       )

        return action_log_probs, actions, entropy, dist, hidden

    def actor_adv_forward(self, input, deterministic=False, evaluate_action=False, normFactor=1, evaluate=False):
        bin_node, next_item, valid_mask, full_mask = \
            DecodeObs4Adv(input, self.bin_node_num, self.next_item_num, self.node_dim)

        invalid_mask = 1 - valid_mask
        valid_length = full_mask.sum(1)
        full_mask = 1 - full_mask

        batch_size = input.size(0)
        bin_node_size = bin_node.size(1)
        next_item_size = next_item.size(1)
        graph_size = bin_node_size + next_item_size

        bin_node = bin_node.contiguous().view(batch_size * bin_node_size, self.node_dim - 2) * normFactor
        next_item = next_item.contiguous().view(batch_size * next_item_size, self.node_dim - 5) * normFactor

        bin_node_embedding = self.init_bin_node_embed(bin_node).reshape((batch_size, -1, self.embedding_size))
        next_item_embedding = self.init_next_item_embed(next_item).reshape((batch_size, -1, self.embedding_size))
        next_item_embedding = self.position_embedder(next_item_embedding) if self.position_embedder else next_item_embedding

        init_embedding = torch.cat((bin_node_embedding, next_item_embedding), dim=1).view(batch_size * graph_size,
                                                                                          self.embedding_size)

        embeddings, _ = self.graph_embedder(init_embedding, mask=full_mask, evaluate=evaluate)
        embedding_shape = (batch_size, graph_size, embeddings.size(-1))

        action_log_probs, actions, entropy, dist, hidden = \
            self._inner(embeddings,
                        deterministic=deterministic,
                        evaluate_action=evaluate_action,
                        shape=embedding_shape,
                        invalid_mask=invalid_mask,
                        full_mask=full_mask,
                        valid_length=valid_length,
                        valid_logit_range=(self.bin_node_num, self.bin_node_num + self.next_item_num)
                        )
        return action_log_probs, actions, entropy, dist, hidden


    def critic_forward(self, input, deterministic=False, evaluate_action=False, normFactor=1, evaluate=False,):

        bin_node, next_item, full_mask = \
            DecodeObs4Critic(input, self.bin_node_num, self.next_item_num, self.node_dim)

        valid_length = full_mask.sum(1)
        full_mask = 1 - full_mask

        batch_size = input.size(0)
        bin_node_size = bin_node.size(1)
        next_item_size = next_item.size(1)
        graph_size = bin_node_size + next_item_size

        bin_node = bin_node.contiguous().view(batch_size * bin_node_size, self.node_dim - 2) * normFactor
        next_item = next_item.contiguous().view(batch_size * next_item_size, self.node_dim - 5) * normFactor

        bin_node_embedding = self.init_bin_node_embed(bin_node).reshape((batch_size, -1, self.embedding_size))
        next_item_embedding = self.init_next_item_embed(next_item).reshape((batch_size, -1, self.embedding_size))
        next_item_embedding = self.position_embedder(next_item_embedding) if self.position_embedder else next_item_embedding

        init_embedding = torch.cat((bin_node_embedding, next_item_embedding,), dim=1).view(
            batch_size * graph_size, self.embedding_size)

        embeddings, _ = self.graph_embedder(init_embedding, mask=full_mask, evaluate=evaluate)
        embedding_shape = (batch_size, graph_size, embeddings.size(-1))

        # (batch_size, graph_size, embed_dim)
        transEmbedding = embeddings.view(embedding_shape)
        # (batch_size, graph_size, embed_dim)
        full_mask = full_mask.view(embedding_shape[0], embedding_shape[1], 1).expand(embedding_shape).bool()
        transEmbedding[full_mask] = 0
        # (batch_size, embed_dim)
        graph_embed = transEmbedding.view(embedding_shape).sum(1)

        # (batch_size, embed_dim) = (batch_size, embed_dim) / (batch_size, 1)
        graph_embed = graph_embed / valid_length.reshape((-1, 1))
        # (batch_size, embed_dim)
        fixed_context = self.project_state_context(graph_embed)

        return fixed_context


    def forward(self, input, deterministic=False, evaluate_action=False,
                normFactor=1, evaluate=False,):

        if self.model_ctg == 'actor':
            if self.policy_ctg == 'place':
                action_log_probs, actions, entropy, dist, hidden = self.actor_place_forward(
                    input, deterministic, evaluate_action, normFactor, evaluate
                )
            elif self.policy_ctg == 'permutation':
                action_log_probs, actions, entropy, dist, hidden = self.actor_adv_forward(
                    input, deterministic, evaluate_action, normFactor, evaluate
                )
            else:
                raise 'Invalid Policy Category'
            return action_log_probs, actions, entropy, dist, hidden

        elif self.model_ctg == 'critic':
            hidden = self.critic_forward(
                input, deterministic, evaluate_action, normFactor, evaluate)
            return hidden

        else:
            pass


    def _inner(self, embeddings, deterministic, evaluate_action, shape, invalid_mask, full_mask, valid_length, valid_logit_range):

        fixed = self._precompute(embeddings, shape=shape, full_mask=full_mask, valid_length=valid_length, )

        prob, invalid_mask = self._get_log_p(fixed, invalid_mask, valid_logit_range=valid_logit_range)

        masked_outs = prob * (1 - invalid_mask) + 1e-20
        prob = torch.div(masked_outs, torch.sum(masked_outs, dim=-1).unsqueeze(1))

        dist = FixedCategorical(probs=prob)
        entropy = dist.entropy()

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if not evaluate_action:
            action_log_probs = dist.log_probs(action)
        else:
            action_log_probs = None

        return action_log_probs, action, entropy, dist, fixed.context_node_projected


    def _get_log_p(self, fixed, invalid_mask, normalize=True, valid_logit_range=None):

        query = fixed.context_node_projected[:, None, :]

        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        logits, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, invalid_mask, valid_logit_range)

        if normalize:
            log_p = torch.log_softmax(logits / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()

        return log_p.exp(), invalid_mask


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, valid_logit_range):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # (self.n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # (self.n_heads, batch_size, num_steps, 1, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # (self.n_heads * batch_size * num_steps, 1, graph_size)
        logits = compatibility.reshape([-1, 1, compatibility.shape[-1]])

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits = logits[:, 0, valid_logit_range[0]:valid_logit_range[1]]
        assert logits.size(-1) == valid_logit_range[1] - valid_logit_range[0]

        if self.mask_logits:
            logits[mask.bool()] = -math.inf

        return logits, None



    def _get_attention_node_data(self, fixed):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key


    def _precompute(self, embeddings, num_steps=1, shape=None, full_mask=None, valid_length=None, ):

        # (batch_size, graph_size, embed_dim)
        transEmbedding = embeddings.view(shape)
        # (batch_size, graph_size, embed_dim)
        full_mask = full_mask.view(shape[0], shape[1], 1).expand(shape).bool()
        transEmbedding[full_mask] = 0
        # (batch_size, embed_dim)
        graph_embed = transEmbedding.view(shape).sum(1)
        # (batch_size * graph_size, embed_dim)
        transEmbedding = transEmbedding.view(-1, embeddings.shape[-1])


        #(batch_size, embed_dim) = (batch_size, embed_dim) / (batch_size, 1)
        graph_embed = graph_embed / valid_length.reshape((-1, 1))
        # (batch_size, embed_dim)
        fixed_context = self.project_fixed_context(graph_embed)

        # (batch_size*graph_size, 3*embed_size) -> (batch_size, 1, graph_size, 3*embed_size)
        # -> 3 * (batch_size, 1, graph_size, embed_size)
        glimpse_key_fixed, glimpse_val_fixed, logit_ley_fixed = \
            self.project_node_embeddings(transEmbedding).view((shape[0], 1, shape[1], -1)).chunk(3, dim=-1)


        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_ley_fixed.contiguous(),)

        # (batch_size * graph_size, embed_dim)
        # (batch_size, embed_dim)
        # (n_heads=1, batch_size, num_step=1, graph_size, embed_size)
        # (n_heads=1, batch_size, num_step=1, graph_size, embed_size)
        # (batch_size, 1, graph_size, embed_size)
        return AttentionModelFixed(transEmbedding, fixed_context, *fixed_attention_node_data)


    def _make_heads(self, v, num_steps=None):
        # (batch_size, 1, graph_size, embed_size) -> (batch_size, 1, graph_size, n_heads, embed_size)
        # -> (batch_size, num_step, graph_size, n_heads, embed_size) ->
        # -> (n_heads=1, batch_size, num_step=1, graph_size, embed_size)

        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        view_shape = (v.size(0), v.size(1), v.size(2), self.n_heads, -1)
        expand_shape = (v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)

        return (v.contiguous().view(*view_shape).expand(*expand_shape).permute(3, 0, 1, 2, 4))


