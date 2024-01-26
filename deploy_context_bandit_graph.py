from rlmd.configuration import configuration
from rlmd.trajectory import trajectory
from rlmd.step import environment
import numpy as np
import multiprocessing
import json
from rlmd.train_graph import ContextBandit
from rlmd.action_space import actions
from rlmd.action_space_v2 import actions as actions_v2
from rlmd.logger import setup_logger
from rgnn.models.reaction_models import PaiNN
from rgnn.models.reaction import ReactionDQN
import torch
from rlmd.action_space import actions
import os
from ase import io

sro = "0.8"
T = 400
kT = T * 8.617 * 10**-5

horizon = 500
model = ReactionDQN.load("model_trained_graph.pt")
trainer = ContextBandit(model, temperature=T)

Tl = []
Cl = []
for u in range(50, 100):
    conf = configuration()
    conf.load("POSCARs/" + sro + "/POSCAR_" + str(u))
    conf.set_potential(platform="mace")
    env = environment(conf, max_iter=100)
    env.relax(accuracy=0.1)

    filename = "POSCARs/" + sro + "/XDATCAR" + str(u)
    io.write(filename, conf.atoms, format="vasp-xdatcar")
    tlist = [0]
    clist = [conf.atoms.get_positions()[-1].tolist()]
    for tstep in range(horizon):
        action_space = actions(conf)
        act_id, act_probs, Q = trainer.select_action(conf.atoms, action_space)
        Gamma = float(torch.sum(torch.exp(Q)))
        dt = 1 / Gamma * 10**-6
        tlist.append(tlist[-1] + dt)
        action = action_space[act_id]
        E_next, fail = env.step(action, accuracy=0.1)
        io.write(filename, conf.atoms, format="vasp-xdatcar", append=True)
        clist.append(conf.atoms.get_positions()[-1].tolist())
        if tstep % 100 == 0:
            print(str(u) + ": " + str(tstep))
    Tl.append(tlist)
    Cl.append(clist)
    with open("POSCARs/" + sro + "/diffuse.json", "w") as file:
        json.dump([Tl, Cl], file)
