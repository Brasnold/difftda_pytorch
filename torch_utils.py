import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from filtrations import rips_percistence_pairs, cubical_persistence_pairs


def torch_rips_filtration_and_persistence(x):
    dist = torch.cdist(x, x)

    with torch.no_grad():
        start_id, end_id = rips_percistence_pairs(dist.numpy())
        start_id, end_id = torch.LongTensor(start_id), torch.LongTensor(end_id)

    start_val = dist[torch.unbind(start_id, 1)]
    end_val = dist[torch.unbind(end_id, 1)]

    pers_pairs = torch.cat([start_val[:, None], end_val[:, None]], 1)

    return pers_pairs


def torch_cubical_filtration_and_persistence(x):
    with torch.no_grad():
        start_id, end_id = cubical_persistence_pairs(x.numpy())
        start_id, end_id = torch.LongTensor(start_id), torch.LongTensor(end_id)

    start_val = x[torch.unbind(start_id, 1)]
    end_val = x[torch.unbind(end_id, 1)]

    pers_pairs = torch.cat([start_val[:, None], end_val[:, None]], 1)

    return pers_pairs

