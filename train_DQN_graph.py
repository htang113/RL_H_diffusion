from rlmd.configuration import configuration
from rlmd.trajectory import trajectory
from rlmd.step import environment
from rlmd.train_graph import Q_trainer
from rlmd.train_graph_v2 import Q_trainer as Q_trainer_v2
from rlmd.action_space import actions
from rlmd.action_space_v2 import actions as actions_v2
from rlmd.action_space_v3 import actions as actions_v3
from rlmd.logger import setup_logger
from rgnn.common.registry import registry
import numpy as np
from ase import io
import os
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

task = "dev/Vrandom_DQN"
horizon = 30
n_traj = 101

species = ["Cr", "Co", "Ni"]

gcnn = registry.get_reaction_model_class("painn").load(
    "/home/hjchun/downloads_git_repo/ReactionGraphNeuralNetwork/dev/best_model_mace_Vrandom_attention.pth.tar"
)
# model = ReactionDQN22(gcnn, N_feat=32)
# target_model = ReactionDQN22(gcnn, N_feat=32)
model = registry.get_model_class("dqn2")(gcnn, N_emb=16, N_feat=32, canonical=True)
target_model = registry.get_model_class("dqn2")(gcnn, N_emb=16, N_feat=32, canonical=True)

pool = ["data/POSCARs_500/POSCAR_" + str(i) for i in range(1, 450)]

traj_list = []
if task not in os.listdir():
    os.makedirs(task, exist_ok=True)
# Configure logging
log_filename = f"{task}/logger.log"  # Define your log filename
opt_log_filenmae = f"{task}" + "/log"
if os.path.exists(opt_log_filenmae):
    os.remove(opt_log_filenmae)

logger = setup_logger("RL", log_filename)
q_params = {
    "temperature": 900,
    "alpha": 0.0,
    "beta": 1.0,
    "dqn": True,
}
trainer = Q_trainer_v2(
    model,
    target_model=target_model,
    logger=logger,
    q_params=q_params,
    lr=5e-5,
    train_all=False,
)

new_pool = []
for filename in pool:
    atoms = io.read(filename)
    if len(atoms) < 500:
        new_pool.append(filename)
logger.info(f"Original pool num: {len(pool)}, Filtered pool num: {len(new_pool)}")

if "traj" not in os.listdir(task):
    os.mkdir(task + "/traj")
if "model" not in os.listdir(task):
    os.mkdir(task + "/model")

with open(task + "/loss.txt", "w") as file:
    file.write("epoch\t loss\n")

for epoch in range(n_traj):
    conf = configuration()
    file = new_pool[np.random.randint(len(new_pool))]
    conf.load(file)
    logger.info("epoch = " + str(epoch) + ":  " + file)
    conf.set_potential(platform="mace")
    env = environment(conf, logfile=task + "/log", max_iter=100)
    env.relax(accuracy=0.1)
    traj_list.append(
        trajectory(q_params["alpha"], q_params["beta"], T=q_params["temperature"])
    )
    for tstep in range(horizon):
        action_space = actions_v3(conf)
        act_id, act_probs, Q = trainer.select_action(conf.atoms, action_space)
        action = action_space[act_id]
        info = {
            "act": act_id,
            "act_probs": act_probs.tolist(),
            "act_space": action_space,
            "state": conf.atoms.copy(),
            "E_min": conf.potential(),
        }

        E_next, fail = env.step(action, accuracy=0.05)
        env.normalize()  # TODO: why do we need is this?
        if not fail and q_params["alpha"] != 0.0:
            E_s, freq, fail = env.saddle(
                action[0], n_points=8, accuracy=0.07, platform="mace"
            )
            info["E_s"], info["log_freq"] = E_s, freq
        elif fail and q_params["alpha"] == 0.0:
            info["E_s"] = 0
            info["log_freq"] = 0
            logger.info("fail step 0")
        else:
            info["E_s"] = 0
            info["log_freq"] = 0

        info["next"], info["fail"], info["E_next"] = conf.atoms.copy(), fail, E_next
        traj_list[-1].add(info)
        if fail:
            logger.info("fail")
        if tstep % 10 == 0 and tstep > 0:
            logger.info("    t = " + str(tstep))

    loss = trainer.update(traj_list, 0.8, (int(1 + epoch ** (2 / 3)), 5), size=1)
    logger.info(f"Epoch: {epoch} | Loss: {loss}")
    with open(task + "/loss.txt", "a") as file:
        file.write(str(epoch) + "\t" + str(loss) + "\n")
    try:
        traj_list[epoch].save(task + "/traj/traj" + str(epoch))
    except:
        logger.info("saving failure")

    if epoch % 10 == 0:
        model.save(task + "/model/model" + str(epoch))
    model.save(task + "/model/model_trained")
