import json
import os

import numpy as np
from ase import io
from rgnn.models.reaction import ReactionDQN2

from rlmd.action_space_v3 import actions as actions_v3
from rlmd.configuration import configuration
from rlmd.logger import setup_logger
from rlmd.step import environment
from rlmd.train_graph import Q_trainer


task = "dev/test_DQN"
if task not in os.listdir():
    os.makedirs(task, exist_ok=True)
model_path = "dev/Vrandom_DQN"
log_filename = f"{task}/logger.log"  # Define your log filename
logger = setup_logger("Deploy", log_filename)
n_episodes = 50
T = 400
horizon = 5
# kT = T * 8.617 * 10**-5

model = ReactionDQN2.load(f"{model_path}/model/model_trained")
target_model = ReactionDQN2.load(f"{model_path}/model/model_trained")
q_params = {
    "temperature": 900,
    "alpha": 0.4,
    "beta": 0.3,
    "dqn": True,
}
trainer = Q_trainer(model=model, logger=logger, q_params=q_params)

new_pool = []
pool = ["data/POSCARs/POSCAR_" + str(i) for i in range(1, 450)]
for filename in pool:
    atoms = io.read(filename)
    if len(atoms) < 500:
        new_pool.append(filename)
logger.info(f"Original pool num: {len(pool)}, Filtered pool num: {len(new_pool)}")

El = []
for u in range(n_episodes):
    conf = configuration()
    file = new_pool[np.random.randint(len(new_pool))]
    conf.load(file)
    conf.set_potential(platform="mace")
    env = environment(conf, max_iter=100, logfile=task + "/log")
    env.relax(accuracy=0.1)

    filename = str(task) + "/XDATCAR" + str(u)
    io.write(filename, conf.atoms, format="vasp-xdatcar")
    Elist = [conf.atoms.get_positions()[-1].tolist()]
    for tstep in range(horizon):
        T = 1000 - 950 * tstep / (horizon - 1)
        trainer.update_T(T)
        action_space = actions_v3(conf)
        act_id, act_probs, Q = trainer.select_action(conf.atoms, action_space)
        action = action_space[act_id]
        E_next, fail = env.step(action, accuracy=0.1)
        io.write(filename, conf.atoms, format="vasp-xdatcar", append=True)
        Elist.append(conf.atoms.get_positions()[-1].tolist())
        if tstep % 100 == 0:
            logger.info(str(u) + ": " + str(tstep))
    El.append(Elist)

    with open(str(task) + "/converge.json", "w") as file:
        json.dump(El, file)
