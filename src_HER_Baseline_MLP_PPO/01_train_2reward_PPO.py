import sys
import os
import threading
import time
from typing import Callable
# 获取当前文件路径
import torch
import argparse
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
print(father_path)
sys.path.append(father_path)    # /src
father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
sys.path.append(father_path)    # /causal-metarl-master
print(father_path)

import numpy as np

# from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from src_HER_Baseline_MLP_PPO.env_do_all_her_2reward import CBNEnv
from stable_baselines3 import PPO
# from src_HER_linear.default import default_config


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        # print("**********************self.locals", self.locals)
        self.logger.record('random_value', value)
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(args):
    current_path = os.path.abspath(__file__)
    father_path1 = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    father_path = os.path.abspath(os.path.dirname(father_path1) + os.path.sep + ".")
    print(args.vertex, args.action_range, args.model_dir)
    env = CBNEnv.create(
        info_phase_length=1440,
        action_range=args.action_range,
        vertex=args.vertex,
        # reward_weight=args.reward_weight,
        reward_scale=args.reward_scale,
        list_last_vertex=[
            # {   # 前1阶段
            #     "vertex":[5],
            #     "dir":"/share/home/liujie/jiangjingchi/yuxuehui/causal-metarl-master-HRL/Model/low_action_5_goal_3_2.zip",
            #     "info_phase_length": 1440,
            #     "action_range":np.inf,
            #     "last_vertex":[3]  },
            # {   # 前2阶段
            #     "vertex":[3],
            #     "dir":father_path + "/Model/her_low_action_3_goal_12_2reward3_copy2.zip",
            #     "info_phase_length": 1440,
            #     "action_range":[-np.inf, np.inf],
            #     "reward_scale":[1.0, 1.0],
            #     "last_vertex":[12]  }
        ],
        n_env=1
    )
    optimizer_kwargs = dict(
        alpha=0.95,
    )
    policy_kwargs = dict(
        optimizer_kwargs=optimizer_kwargs,
        optimizer_class=RMSpropTFLike
    )
    model = PPO("MlpPolicy",
                env,
                n_steps=args.n_step,
                gae_lambda=0.95,
                gamma=0.9,
                n_epochs=10,
                ent_coef=0.0,
                learning_rate=float(args.lr),
                clip_range=0.2,
                use_sde=True,
                sde_sample_freq=4,
                verbose=1,
                tensorboard_log='../logs',
                seed=args.seed
                )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=TensorboardCallback()
    )

    path = os.path.join(    # 这里写的是存储模型的路径
        father_path,
        'Model_Baseline_x10_x12',
        args.model_dir,
    )
    print("Save Model to path:", path)
    model.save(path)


#
if __name__ == "__main__":
    # 创建一个参数解析实例
    parser = argparse.ArgumentParser()
    # 添加参数解析
    parser.add_argument("-vertex", nargs='+', type=int, help="Now inserve")
    parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, 3000])    # 如果步输入的话就是np.inf
    parser.add_argument("-lr", type=float, help=" ", default=float(1e-4))  # 如果步输入的话就是1.0
    parser.add_argument("-model_dir", type=str, help="Save dir")
    parser.add_argument("-reward_scale", nargs='+', type=float, help=" ", default=[1.0, 1.0])
    parser.add_argument("-n_step", type=int, help="n_step", default=5)
    parser.add_argument("-seed", type=int, help="seed", default=94566)
    parser.add_argument("-max_grad_norm", type=float, help="seed", default=10000)
    parser.add_argument("-total_timesteps", type=int, help="seed", default=int(3e7))
    # 开始解析
    args = parser.parse_args()

    train(args)
