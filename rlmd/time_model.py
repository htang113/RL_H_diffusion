#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:08:57 2024

@author: ubuntu
"""

from abc import ABC
from typing import Any, Dict, Literal, Tuple, List

import torch
from torch.nn import functional as F
from torch_geometric.utils import scatter
from torch_geometric.utils import degree
from rgnn.common import keys as K
from rgnn.common.registry import registry
from rgnn.common.typing import DataDict, Tensor
from rgnn.common.configuration import Configurable
from rgnn.graph.utils import compute_neighbor_vecs
from rgnn.models.nn.mlp import MLP
from rgnn.models.nn.painn.representation import PaiNNRepresentation
from rgnn.models.nn.scale import PerSpeciesScaleShift, canocialize_species


@registry.register_model("p_net_v2")
class PNet2(torch.nn.Module, Configurable, ABC):
    """PaiNN model, as described in https://arxiv.org/abs/2102.03150.
    This model applies equivariant message passing layers within cartesian coordinates.
    Provides "node_features" and "node_vec_features" embeddings.

    Args:
        species (list[str]): List of atomic species to consider.
        cutoff (float): Cutoff radius for interactions. Defaults to 5.0.
        hidden_channels (int): Number of hidden channels in the convolutional layers. Defaults to 128.
        n_interactions (int): Number of message passing layers. Defaults to 3.
        rbf_type (str): Type of radial basis functions. One of "gaussian" or "bessel".
            Defaults to "bessel".
        n_rbf (int): Number of radial basis functions. Defaults to 20.
        trainable_rbf (bool): Whether to train the radial basis functions. Defaults to False.
        activation (str): Activation function to use in the convolutional layers. Defaults to "silu".
        shared_interactions (bool): Whether to share the convolutional layers across interactions.
            Defaults to False.
        shared_filters (bool): Whether to share the convolutional filters across interactions.
            Defaults to False.
        epsilon (float): Small value to add to the denominator for numerical stability. Defaults to 1e-8.
    """

    embedding_keys = [K.node_features]
    # embedding_keys = [K.node_features, K.node_vec_features, K.reaction_features]

    def __init__(
        self,
        species,
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        n_interactions: int = 3,
        rbf_type: Literal["gaussian", "bessel"] = "bessel",
        n_rbf: int = 20,
        trainable_rbf: bool = False,
        activation: str = "silu",
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        # canonical: bool = False,
        # N_emb: int = 16,
        n_feat: int = 32,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        atomic_numbers = [val.item() for val in canocialize_species(species)]
        atomic_numbers_dict = {}
        for i, key in enumerate(atomic_numbers):
            atomic_numbers_dict.update({key: species[i]})
        atomic_numbers.sort()
        self.atomic_numbers = atomic_numbers
        sorted_species = []
        for n in atomic_numbers:
            sorted_species.append(atomic_numbers_dict[n])
        self.species = sorted_species
        self.cutoff = cutoff
        self.species_energy_scale = PerSpeciesScaleShift(species)
        self.embedding_keys = self.__class__.embedding_keys
        # TODO: Should be more rigorous
        self.hidden_channels = hidden_channels
        self.n_interactions = n_interactions
        self.rbf_type = rbf_type
        self.n_rbf = n_rbf
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.shared_interactions = shared_interactions
        self.shared_filters = shared_filters
        self.epsilon = epsilon
        # self.canonical = self.canonical
        self.dropout_rate = dropout_rate
        self.n_feat = n_feat

        self.representation = PaiNNRepresentation(
            hidden_channels=hidden_channels,
            n_interactions=n_interactions,
            rbf_type=rbf_type,
            n_rbf=n_rbf,
            trainable_rbf=trainable_rbf,
            cutoff=cutoff,
            activation=activation,
            shared_interactions=shared_interactions,
            shared_filters=shared_filters,
            epsilon=epsilon,
        )
        self.probs_out = MLP(
            n_input=hidden_channels + 1,
            n_output=1,
            hidden_layers=(self.n_feat, self.n_feat),
            activation="silu",
            w_init="xavier_uniform",
            b_init="zeros",
        )
        self.kb = 8.617 * 10**-5
        self.reset_parameters()

    def reset_parameters(self):
        self.representation.reset_parameters()
        self.probs_out.reset_parameters()

    def forward(self, data: DataDict):
        compute_neighbor_vecs(data)
        node_degrees = degree(data["edge_index"][0], num_nodes=torch.sum(data["n_atoms"]))
        new_features_list = []
        # new_features_list_2 = []
        for graph_id in torch.unique(data["batch"]):
            mask = data["batch"] == graph_id
            filtered_nodes = node_degrees[mask]

            _, sorted_index = torch.sort(filtered_nodes, dim=-1, stable=True)
            new_features_list.append(sorted_index)
        connectivity_feat = torch.concat(new_features_list, dim=0)
        data.update({K.elems: connectivity_feat})
        data = self.representation(data)

        time_components = self.probs_out(
            torch.cat([data["kT"].unsqueeze(-1), data[K.node_features]], dim=-1)
        ).squeeze(-1)

        return torch.mean(time_components);

    @torch.jit.unused
    def save(self, filename: str):
        state_dict = self.state_dict()
        hyperparams = self.get_config()
        state = {"state_dict": state_dict, "hyper_parameters": hyperparams}

        torch.save(state, filename)

    # "best_metric": best_metric,
    @torch.jit.unused
    @classmethod
    def load(cls, path: str):
        """Load the model from checkpoint created by pytorch lightning.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            InterAtomicPotential: The loaded model.
        """
        map_location = None if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=map_location)
        hparams = ckpt["hyper_parameters"]
        hparams.pop("@name")
        state_dict = ckpt["state_dict"]
        model = cls.from_config(hparams)
        model.load_state_dict(state_dict=state_dict)
        return model