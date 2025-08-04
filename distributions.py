import torch
import torch.nn as nn
from torch.distributions import Normal, SigmoidTransform, AffineTransform, TransformedDistribution

"""
Modify standard PyTorch distributions so they are compaible with this code.
"""

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()


class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale, high_value):
        self.loc = loc
        self.scale = scale

        self.base_dist = Normal(loc, scale)
        transforms = [SigmoidTransform(),
                      AffineTransform(loc=0., scale=high_value)]
        super(SquashedNormal, self).__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for trans in self.transforms:
            mu = trans(mu)
        return mu

    @property
    def mode(self):
        return self.mean

    def log_probs(self, actions):
        return self.log_prob(actions).sum(-1, keepdim=True)



if __name__ == '__main__':
    cate = FixedCategorical(probs=torch.Tensor([[0.8, 0.2], [0.1, 0.9]]))
    sample = cate.sample()
    log_p = cate.log_probs(sample)
    #print(sample, log_p.exp())

    cons = torch.ones((2, 5, 3, 4))
    cons = cons[torch.arange(sample.size(0)), sample.squeeze(-1), ...]
    #print(cons.size())

    mean = nn.Parameter(torch.ones((2, 1)) * 5.)
    var = nn.Parameter(torch.ones((2, 1)) * 2)

    dist = SquashedNormal(mean, var, 10.)

    sample = torch.Tensor([[[5], [2], [8]], [[2], [8], [5]]])
    log_p = dist.log_prob(sample.squeeze(-1)).exp()

    norm_p = torch.div(log_p, torch.sum(log_p, dim=-1).unsqueeze(-1))

    max_index = torch.argmax(norm_p, dim=-1)
    act = sample[torch.arange(max_index.size(0)), max_index, ...]

    max_index = torch.multinomial(norm_p, num_samples=1).squeeze(-1)
    act = sample[torch.arange(max_index.size(0)), max_index, ...]
    print(act)

    log_p = dist.log_prob(act)
    log_p.sum().backward()
    print(mean.grad.data)