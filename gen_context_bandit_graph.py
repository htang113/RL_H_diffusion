from rlmd.configuration import configuration
from rlmd.trajectory import trajectory
from rlmd.step import environment
from rlmd.train_graph import ContextBandit
from rlmd.action_space import actions
from rlmd.action_space_v2 import actions as actions_v3
from rlmd.logger import setup_logger
from rgnn.models.reaction_models import PaiNN
from rgnn.models.reaction import ReactionDQN
import numpy as np

import os
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

task = "dev/V_Diff/traj0"
horizon = 20
n_traj = 100

species = ["H", "Cr", "Co", "Ni"]

# gcnn = PaiNN(species=species)
reaction_model = ReactionDQN.load("best_model.pth.tar")
trainer = ContextBandit(reaction_model, temperature=1000, lr=5e-5)

pool = ["POSCARs/CONTCAR_H_CCN" + str(i) for i in range(1, 450)]

traj_list = []
if task not in os.listdir():
    os.makedirs(task, exist_ok=True)
# Configure logging
log_filename = f"{task}/logger.log"  # Define your log filename
opt_log_filenmae = f"{task}" + "/log"
if os.path.exists(opt_log_filenmae):
    os.remove(opt_log_filenmae)

logger = setup_logger("RL", log_filename)


if "traj" not in os.listdir(task):
    os.mkdir(task + "/traj")
if "model" not in os.listdir(task):
    os.mkdir(task + "/model")

with open(task + "/loss.txt", "w") as file:
    file.write("epoch\t loss\n")

for epoch in range(n_traj):
    conf = configuration()
    file = pool[np.random.randint(len(pool))]
    conf.load(file)
    logger.info("epoch = " + str(epoch) + ":  " + file)
    conf.set_potential()
    env = environment(conf, logfile=task + "/log", max_iter=100)
    env.relax(accuracy=0.1)
    traj_list.append(trajectory(1, 0))
    for tstep in range(horizon):
        action_space = actions(conf, dist_mul_body=1.2, act_mul=1.6, act_mul_move=1.2)
        act_id = np.random.choice(len(action_space))
        action = action_space[act_id]
        info = {
            "act": act_id,
            "act_probs": [],
            "act_space": action_space,
            "state": conf.atoms.copy(),
            "E_min": conf.potential(),
        }

        E_next, fail = env.step(action, accuracy=0.05)
        if not fail:
            E_s, freq, fail = env.saddle(action[0], n_points=8, accuracy=0.07)
            info["E_s"], info["log_freq"] = E_s, freq
        else:
            info["E_s"] = 0
            info["log_freq"] = 0
            logger.info("fail step 1")

        info["next"], info["fail"], info["E_next"] = conf.atoms.copy(), fail, E_next
        traj_list[-1].add(info)
        if fail:
            logger.info("fail")
        if tstep % 10 == 0 and tstep > 0:
            logger.info("    t = " + str(tstep))

    with open(task + "/loss.txt", "a") as file:
        file.write(str(epoch) + "\t" + str(loss) + "\n")
    try:
        traj_list[epoch].save(task + "/traj/traj" + str(epoch))
    except:
        logger.info("saving failure")
