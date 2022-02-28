import copy

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from models.GCNLayer import GraphConvolution
from models.HighWay import HighWay


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class GCN(nn.Module):
    def __init__(self, hyps, mutual_link, device=torch.device("cpu")):
        super(GCN, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.device = device
        self.mutual_link = mutual_link

        # GCN
        self.gcns = nn.ModuleList()
        for i in range(hyps["gcn_layers"]):
            gcn = GraphConvolution(in_features=hyps["in_features"] if i != 0 else hyps["out_features"],
                                   out_features=hyps["out_features"],
                                   edge_types=hyps["edge_types"],
                                   dropout=hyps["gcn_dp"] if i != hyps["gcn_layers"] - 1 else None,
                                   use_bn=hyps["gcn_use_bn"],
                                   device=device)
            self.gcns.append(gcn)

        # Highway
        if hyps["use_highway"]:
            if hyps["in_features"] == hyps["out_features"]:
                self.hws = nn.ModuleList()
                for i in range(hyps["gcn_layers"]):
                    hw = HighWay(size=hyps["out_features"], dropout_ratio=hyps["gcn_dp"])
                    self.hws.append(hw)
            else:
                print("When using highway, the input feature size should be equivalent to the output feature size. "
                      "The highway structure is abandoned.")

        self.W = nn.ParameterList()

        w = nn.Parameter(torch.Tensor(hyps["in_features"], hyps["out_features"]))
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('tanh'))
        self.W.append(w)

        for i in range(hyps["gcn_layers"]):
            w = nn.Parameter(torch.Tensor(hyps["out_features"], hyps["out_features"]))
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('tanh'))
            self.W.append(w)

        self.to(self.device)


    def forward(self, seq_features1, mask1, seq_features2, mask2, adj):
        max_seq1_len = seq_features1.size(1)
        seq_features = torch.cat((seq_features1, seq_features2), 1)
        mask = torch.matmul(torch.unsqueeze(mask1, 2), torch.unsqueeze(mask2, 1))

        for i in range(self.hyperparams["gcn_layers"]):

            seq_features1 = seq_features[:, :max_seq1_len, :]
            seq_features2 = seq_features[:, max_seq1_len:, :]

            C = F.tanh(torch.matmul(torch.matmul(seq_features1, self.W[i]), torch.transpose(seq_features2, 1, 2)))

            if self.mutual_link == "co_attn":
                C1 = masked_softmax(C, mask)
                C2 = masked_softmax(torch.transpose(C, 1, 2), torch.transpose(mask, 1, 2))

                adj_mask = torch.zeros(adj.size(0), adj.size(1) - 1, adj.size(2), adj.size(3)).to(self.device)  # edge_types == adj.size(1)
                C1 = torch.cat((torch.zeros(C1.size(0), C1.size(1), adj.size(3) - C1.size(2)).to(self.device), C1), 2)
                C2 = torch.cat((C2, torch.zeros(C2.size(0), C2.size(1), adj.size(3) - C2.size(2)).to(self.device)), 2)
                adj_mask = torch.cat((adj_mask, torch.unsqueeze(torch.cat((C1, C2), 1), 1)), 1)

                adj_mask = adj_mask + adj
            else:
                adj_mask = adj

            seq_features = torch.cat((seq_features1, seq_features2), 1)

            if self.hyperparams["use_highway"]:
                seq_features = self.gcns[i](seq_features, adj_mask) + self.hws[i](
                    seq_features)  # (batch_size, seq_len, d')
            else:
                seq_features = self.gcns[i](seq_features, adj_mask)

        seq_features1 = seq_features[:, :max_seq1_len, :]
        seq_features2 = seq_features[:, max_seq1_len:, :]

        return seq_features1, seq_features2
