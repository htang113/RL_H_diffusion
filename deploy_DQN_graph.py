import json
import os

import numpy as np
from ase import io
from rgnn.common.registry import registry

from rlmd.action_space_v3 import actions as actions_v3
from rlmd.configuration import configuration
from rlmd.logger import setup_logger
from rlmd.step import environment
from rlmd.train_graph_v2 import select_action


task = "dev/DQN_800"
if task not in os.listdir():
    os.makedirs(task, exist_ok=True)
model_path = "dev/Vrandom_DQN"
log_filename = f"{task}/logger.log"  # Define your log filename
logger = setup_logger("Deploy", log_filename)
n_episodes = 10
T_start = 1200
T_end = 800
horizon = 200
# kT = T * 8.617 * 10**-5

model = registry.get_model_class("dqn").load_old(f"{model_path}/model/model_trained")
target_model = registry.get_model_class("dqn").load_old(
    f"{model_path}/model/model_trained"
)
q_params = {
    "alpha": 0.0,
    "beta": 1.0,
    "dqn": True,
}
# trainer = Q_trainer(model=model, logger=logger, q_params=q_params)

new_pool = []
pool = ["data/POSCARs_500/POSCAR_" + str(i) for i in range(1, 450)]
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
    logger.info(f"Episode: {u}")
    for tstep in range(horizon):
        T = T_start - (T_start - T_end) * tstep / horizon
        q_params.update({"temperature": T})
        action_space = actions_v3(conf)
        act_id, act_probs, Q = select_action(model, conf.atoms, action_space, q_params)
        action = action_space[act_id]
        E_next, fail = env.step(action, accuracy=0.1)
        io.write(filename, conf.atoms, format="vasp-xdatcar", append=True)
        energy = conf.potential()
        Elist.append(energy)
        if tstep % 10 == 0:
            logger.info(
                f"Step: {tstep}, T: {q_params['temperature']:.3f}, E: {energy:.3f}"
            )

    with open(str(task) + "/converge.json", "w") as file:
        json.dump(El, file)
