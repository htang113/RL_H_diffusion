#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:20:17 2024

@author: ubuntu
"""

from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import torch
import torch.nn.functional as F
from rgnn.graph.utils import batch_to
from rgnn.graph.atoms import AtomsGraph
from rgnn.graph.dataset.atoms import AtomsDataset
from torch.nn import functional as F
from torch_geometric.loader import DataLoader


def make_p_data(atoms, q_params, cutoff) -> AtomsGraph:

    data = AtomsGraph.from_ase(atoms,
                               cutoff,
                               read_properties=False,
                               neighborlist_backend="ase",
                               add_batch=True)
    data.kT = torch.repeat_interleave(
        torch.as_tensor([q_params["temperature"] * 8.617 * 10**-5],
                        dtype=torch.get_default_dtype(),
                        device=data["elems"].device), data["n_atoms"])

    return data


def make_p_dataset(atoms_list, target_time_list,
                   q_params, cutoff) -> AtomsDataset:
    dataset_list = []
    # target_p_tensor = pad_tensor_list(target_p_list)
    for i, atoms in enumerate(atoms_list):
        data = make_p_data(atoms, q_params, cutoff);
        data.time = torch.tensor(target_time_list[i]).unsqueeze(0);
        dataset_list.append(data);

    dataset = AtomsDataset(dataset_list)

    return dataset