# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:40:00 2022

@author: 17000
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase import Atoms
from rgnn.graph.dataset.reaction import ReactionDataset
from rgnn.graph.reaction import ReactionGraph
from rgnn.graph.utils import batch_to
from rgnn.models.reaction import ReactionGNN
from rgnn.models.reaction_models import PaiNN
from torch_geometric.loader import DataLoader
from rgnn.train.loss import WeightedSumLoss
from rgnn.models.reaction_models.utils import get_scaler


class ContextBandit:
    def __init__(self, model: ReactionGNN, lr=10**-3, eps_clip=0.01, temperature=300):
        self.policy_value_net = model
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.eps_clip = eps_clip
        self.kT = temperature * 8.617 * 10**-5

    def select_action(self, state, action_space, device="cuda"):
        self.policy_value_net.to(device)
        dataset = convert(state, action_space)
        total_q_list = []
        dataloader = DataLoader(dataset, batch_size=16)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch_to(batch, device)
                rl_q = self.policy_value_net.get_q(
                    batch, temperature=self.kT, alpha=1.0, beta=0.0
                )["rl_q"]
                total_q_list.append(rl_q.detach())
        Q = torch.concat(total_q_list, dim=-1)
        action_probs = nn.Softmax(dim=0)(Q)
        action = np.random.choice(
            len(action_probs.detach().cpu().numpy()),
            p=action_probs.detach().cpu().numpy(),
        )

        return action, action_probs, Q

    def update(self, memory_l, gamma, steps, batch_size=8, device="cuda"):
        self.policy_value_net.to(device)
        ave_loss = 0
        loss_fn = WeightedSumLoss(
            keys=("barrier", "freq"),
            weights=(1.0, self.kT**2),
            loss_fns=("mse_loss", "mse_loss"),
        )
        # loss_fn = WeightedSumLoss(keys=("q0", "q1"),weights=(1.0, self.kT**2), loss_fns=("mse_loss", "mse_loss"))

        for m in range(steps[0]):
            prob = [0.99 ** (len(memory_l) - i) for i in range(len(memory_l))]
            randint = np.random.choice(range(len(memory_l)), p=prob / np.sum(prob))
            memory = memory_l[randint]
            states = memory.states
            aspace = memory.act_space
            actions = memory.actions
            taken_actions = [[aspace[i][actions[i]]] for i in range(len(aspace))]
            rewards = torch.tensor(memory.rewards, dtype=torch.float)
            freq = torch.tensor(memory.freq, dtype=torch.float)
            delta_e = torch.tensor(
                [val - memory.E_min[i] for i, val in enumerate(memory.E_next)],
                dtype=torch.float,
            )

            total_dataset_list = []
            for i, state in enumerate(states):
                dataset = convert(state, taken_actions[i])
                total_dataset_list += dataset

            total_dataset_added_list = []
            for i, data in enumerate(total_dataset_list):
                data.q1 = torch.tensor([freq[i]])
                data.q0 = torch.tensor([rewards[i]])
                data.delta_e = torch.tensor([delta_e[i]])
                data.freq = torch.tensor([freq[i]])
                data.barrier = torch.tensor([-rewards[i]])
                total_dataset_added_list.append(data)

            total_dataset = ReactionDataset(total_dataset_added_list)
            # TODO: Should be remove for the static model
            # means, stddevs = get_scaler(["barrier", "freq", "delta_e"], total_dataset)
            # self.policy_value_net.reaction_model.means = means
            # self.policy_value_net.reaction_model.stddevs = stddevs

            q_dataloader = DataLoader(total_dataset, batch_size=batch_size)
            for epoch in range(steps[1]):
                for batch in q_dataloader:
                    batch = batch_to(batch, device)
                    output = self.policy_value_net(batch)
                    q_loss = loss_fn(output, batch)
                    self.optimizer.zero_grad()
                    q_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_value_net.parameters(), 1.0
                    )
                    self.optimizer.step()
                ave_loss += q_loss.detach().item()
                print(f"* Epoch {epoch},Q loss: {q_loss.detach().item()}")

        return ave_loss / steps[0] / steps[1]


class Q_trainer:
    def __init__(self, model, target, lr=10**-3, temperature=300):
        self.policy_value_net = model
        self.target_net = target
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.kT = temperature * 8.617 * 10**-5

    def select_action(self, state, action_space):
        self.policy_value_net.convert([state], [action_space])
        Q = self.policy_value_net()[0]
        action_probs = nn.Softmax(dim=0)(Q[:, 0] / self.kT)
        action = np.random.choice(
            len(action_probs.detach().numpy()), p=action_probs.detach().numpy()
        )

        return action, action_probs, Q

    def update(self, memory_l, gamma, steps, size=10):
        self.target_net.load_state_dict(self.policy_value_net.state_dict())
        ave_loss = 0
        for m in range(steps[0]):
            randint = np.random.randint(len(memory_l), size=size)
            states, next_states, taken_actions, rewards, next_aspace = (
                [],
                [],
                [],
                [],
                [],
            )
            for u in randint:
                memory = memory_l[u]
                states += memory.states[:-1]
                next_states += memory.next_states[:-1]
                rewards += memory.rewards[:-1]
                aspace = memory.act_space
                actions = memory.actions
                taken_actions += [
                    [aspace[i][actions[i]]] for i in range(len(aspace) - 1)
                ]
                next_aspace += [aspace[i] for i in range(1, len(aspace))]
            rewards = torch.tensor(rewards, dtype=torch.float)
            self.policy_value_net.convert(states, taken_actions)
            next_Q = torch.zeros(len(next_aspace))
            for i in range(len(next_aspace)):
                self.target_net.convert([next_states[i]], [next_aspace[i]])
                next_Q[i] = torch.max(self.target_net())

            for epoch in range(steps[1]):
                Q = self.policy_value_net()
                target = next_Q * gamma + rewards
                loss = torch.mean((target - Q[:, 0, 0]) ** 2)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                ave_loss += loss

        return ave_loss / steps[0] / steps[1]


def convert(atoms: Atoms, actions: List[List[float]]) -> ReactionDataset:
    traj_reactant = []
    traj_product = []
    for act in actions:
        traj_reactant.append(atoms)
        final_atoms = atoms.copy()
        final_positions = []
        for i, pos in enumerate(final_atoms.get_positions()):
            if i == act[0]:
                new_pos = pos + act[1:]
                final_positions.append(new_pos)
            else:
                final_positions.append(pos)
        final_atoms.set_positions(final_positions)
        traj_product.append(final_atoms)
    dataset_list = []
    for i in range(len(traj_reactant)):
        data = ReactionGraph.from_ase(traj_reactant[i], traj_product[i])
        dataset_list.append(data)
    dataset = ReactionDataset(dataset_list)

    return dataset
