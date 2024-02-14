import json
import os

import numpy as np
import torch
from ase import io
from rgnn.models.reaction import ReactionDQN2

from rlmd.action_space_v3 import actions as actions_v3
from rlmd.configuration import configuration
from rlmd.logger import setup_logger
from rlmd.step import environment
from rlmd.train_graph import ContextBandit

task = "dev/test_CB"
if task not in os.listdir():
    os.makedirs(task, exist_ok=True)
model_path = "dev/Vrandom_CB"
log_filename = f"{task}/logger.log"  # Define your log filename
logger = setup_logger("Deploy", log_filename)
# sro = "0.8"
n_episodes = 30
T = 400
# kT = T * 8.617 * 10**-5

horizon = 5
model = ReactionDQN2.load(f"{model_path}/model/model_trained")
trainer = ContextBandit(model, logger=logger, temperature=T)

Tl = []
Cl = []
pool = ["data/POSCARs/POSCAR_" + str(i) for i in range(1, 450)]
new_pool = []
for filename in pool:
    atoms = io.read(filename)
    if len(atoms) < 500:
        new_pool.append(filename)
logger.info(f"Original pool num: {len(pool)}, Filtered pool num: {len(new_pool)}")

for u in range(n_episodes):
    conf = configuration()
    file = new_pool[np.random.randint(len(new_pool))]
    conf.load(file)
    conf.set_potential(platform="mace")
    env = environment(conf, max_iter=100, logfile=task + "/log")
    env.relax(accuracy=0.1)

    filename = str(task) + "/XDATCAR" + str(u)
    io.write(filename, conf.atoms, format="vasp-xdatcar")
    tlist = [0]
    clist = [conf.atoms.get_positions()[-1].tolist()]
    for tstep in range(horizon):
        action_space = actions_v3(conf)
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
    with open(str(task) + "/diffuse.json", "w") as file:
        json.dump([Tl, Cl], file)
