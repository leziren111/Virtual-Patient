from collections import defaultdict
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import copy

# from src.causal import CausalGraph
from src_HER_Baseline_MLP_PPO.insulin_causal_x4_x8_simpler import CausalGraph
import torch
import torch.nn.functional as F
from src_HER_Baseline_MLP_PPO.env_do_all_2reward import PreCBN
from src.a2c import A2C

import operator
from functools import reduce


def one_hot(length, idx):
    one_hot = np.zeros(length)
    one_hot[idx] = 1
    return one_hot




class CBNEnv(Env):
    def __init__(self,
                 agent_type='default',
                 info_phase_length=1440,
                 action_range=[-np.inf, np.inf],
                 vertex=[5],
                 reward_scale = [1.0, 1.0],
                 list_last_vertex=[],
                 train=True,
                 ):
        """
        Create a stable_baselines-compatible environment to train policies on
        :param agent_type: 'default'；不可修改
        :param info_phase_length: episode的最大步长，注意不同阶段最大步长不同；在create函数中作为超参转入；
        :param vertex: action干预的node的编号
        :param list_last_vertex: List格式，其中元素为Dict格式；
                            前序阶段的agent信息，用来生成本阶段的goal；
                            若为x3-x12阶段（即没有前序阶段），则为空[];
                            反序的！例：x10-x5(当前阶段),x5-x3,x3-x12;
                            生成goal的时候是从后往前遍历，逐阶段生成goal;
        :param train:
        """
        # vg = 1.886280201
        self.logger = None
        self.log_data = defaultdict(int)
        self.agent_type = agent_type
        self.ep_rew = 0
        self.reward_scale = reward_scale  # 用来平衡 r1和r2
        self.vertex = vertex  # 这一次干预的顶点(list),即第几个顶点，为一个列表值
        self.list_last_vertex = list_last_vertex
        self.reward_env = []  # 生成env列表
        self.reward_goal = []  # 生成goal列表
        self.reward_state = []  # 记录各个阶段 当前时刻 状态信息
        for dic in list_last_vertex:  # 注意last_vertex为反向的
            if 12 in dic["last_vertex"]:
                env1 = PreCBN.create(goal=[[70*1.886280201, 180*1.886280201]], vertex=dic["vertex"], last_vertex=dic["last_vertex"], info_phase_length=dic["info_phase_length"], reward_scale=dic["reward_scale"], action_range=dic["action_range"], n_env=1)
                #  create(cls, goal, vertex, last_vertex, info_phase_length, action_range, n_env):
                model1 = A2C.load(dic["dir"], env=env1)
                obs1 = env1.envs[0].reset()
                self.reward_env.append(copy.deepcopy(env1))
                self.reward_goal.append(copy.deepcopy(model1))
                self.reward_state.append(copy.deepcopy(obs1))
            else:  # 需要计算goal
                # 1、首先使用reward_goal[0]，根据state生成action，即x3的值，然后取 -1,1的波动
                # 2、修改env的goal
                env1 = PreCBN.create(goal=[[70*1.886280201, 180*1.886280201]]*len(dic["last_vertex"]), vertex=dic["vertex"], last_vertex=dic["last_vertex"], info_phase_length=dic["info_phase_length"], reward_scale=dic["reward_scale"], action_range=dic["action_range"], n_env=1)
                # print(env1.envs[0].observation_space)
                # ！！！ 注意这里先用goal=[[70, 180]] 暂时初始化环境，之后再修改;
                # ！！！ [[70, 180]]*len(dic["last_vertex"])是因为可能有多个目标node
                model1 = A2C.load(dic["dir"], env=env1)
                obs1 = env1.envs[0].reset()
                self.reward_env.append(copy.deepcopy(env1))
                self.reward_goal.append(copy.deepcopy(model1))
                self.reward_state.append(copy.deepcopy(obs1))

        # 初始化所有阶段的goal
        if len(list_last_vertex) > 0:
            self.last_vertex = list_last_vertex[0]["vertex"]  # 上一次干预的顶点
            # 重新采样goal
            goal_temp = []
            for idx in range(len(self.reward_goal) - 1, -1, -1):  # 逆序遍历
                if idx == len(self.reward_goal) - 1:  # 最后一个
                    action_temp, _states = self.reward_goal[idx].predict(self.reward_state[idx])    # 获得动作
                    obs_temp, _rewards, _dones, _info = self.reward_env[idx].envs[0].step(action_temp)  # env step
                    self.reward_state[idx] = copy.deepcopy(obs_temp)      # 记录当前阶段，当前时刻状态
                    goal_temp = []
                    for item in self.reward_env[idx].envs[0].vertex:  # 存储goal，下个阶段的model需要使用
                        # ！！！ 因为action为增量，所以上述操作目的是取node 的值， 即最后做reward的时候是 目标node 当前值 与 目标值 之间的差距
                        goal_temp.append([self.reward_env[idx].envs[0].state.get_value(item) - 10.0,
                                          self.reward_env[idx].envs[0].state.get_value(item) + 10.0])

                else:
                    self.reward_env[idx].envs[0].goal = copy.deepcopy(goal_temp)    # 修改当前goal
                    action_temp, _states = self.reward_goal[idx].predict(self.reward_state[idx])
                    obs_temp, _rewards, _dones, _info = self.reward_env[idx].envs[0].step(action_temp)
                    self.reward_state[idx] = copy.deepcopy(obs_temp)
                    goal_temp = []
                    for item in self.reward_env[idx].envs[0].vertex:  # 存储goal，下个阶段的model需要使用
                        goal_temp.append([self.reward_env[idx].envs[0].state.get_value(item) - 10.0,
                                          self.reward_env[idx].envs[0].state.get_value(item) + 10.0])
            self.goal = copy.deepcopy(goal_temp)    # 修改当前阶段的 goal

        else:
            # 当前阶段为x3-x12，last_vertex 为空
            self.last_vertex = [12]  # 上一次干预的顶点
            self.goal = [[70*1.886280201, 180*1.886280201]]

        if train:
            self.state = TrainEnvState(self.vertex, self.last_vertex, info_phase_length)  # 一个继承了 EnvState class 的类
        else:
            self.state = TestEnvState(self.vertex, self.last_vertex, info_phase_length)
        # self.action_space = Box(action_range[0], action_range[1], (len(self.vertex),), dtype=np.float64)
        self.action_space = Box(action_range[0], action_range[1], (1,), dtype=np.float64)
        self.observation_space = Box(-np.inf, np.inf, (self.state.graph.len_obe + 2 * len(self.goal),))  # 得到子图的节点数
        print("*************************", self.observation_space)

    @classmethod
    def create(cls, info_phase_length, vertex, reward_scale, list_last_vertex, action_range, n_env):
        return DummyVecEnv([lambda: cls(info_phase_length=info_phase_length, vertex=vertex, reward_scale=reward_scale, list_last_vertex=list_last_vertex, action_range=action_range)
                            for _ in range(n_env)])

    def reward(self, val, observed_valse):
        """
        reward需要根据goal计算
        :param val:
        :return:
        """

        if len(val) == 1:
            max_state = torch.from_numpy(np.array(val))  # 转为tensor
            score_offset = abs(F.relu(self.goal[0][0] - max_state[0]) + F.relu(max_state[0] - self.goal[0][1]))
            score_center_plus = abs(
                min(self.goal[0][1], max(self.goal[0][0], max_state[0])) - (self.goal[0][0] + self.goal[0][1]) / 2)
            score_center_plus = 24 - score_offset - 0.2 * score_center_plus

        elif len(val) == 2:
            max_state = torch.from_numpy(np.array(val))
            score_offset = abs(F.relu(self.goal[0][0] - max_state[0]) + F.relu(max_state[0] - self.goal[0][1]))
            score_center_plus = abs(
                min(self.goal[0][1], max(self.goal[0][0], max_state[0])) - (self.goal[0][0] + self.goal[0][1]) / 2)
            score_center_plus = 24 - score_offset - 0.2 * score_center_plus
            # 第一个被干预节点的reward
            score_offset_ = abs(F.relu(self.goal[1][0] - max_state[1]) + F.relu(max_state[1] - self.goal[1][1]))
            score_center_plus_ = abs(
                min(self.goal[1][1], max(self.goal[1][0], max_state[1])) - (self.goal[1][0] + self.goal[1][1]) / 2)
            score_center_plus_ = 24 - score_offset_ - 0.2 * score_center_plus_
            # 第二个被干预节点的reward
            score_center_plus += score_center_plus_  # 两者相加

        if (self.state.info_steps >= 360 and self.state.info_steps <= 420 ) \
            or (self.state.info_steps >= 660 and self.state.info_steps <= 720) \
            or (self.state.info_steps >= 1080 and self.state.info_steps <= 1140 ):
            # 用餐时间消除波动惩罚
            r2 = 0.0
        else:
            temp_reward = 0
            if 4 in self.vertex and 8 in self.vertex:
                temp_reward = 4
            elif (7 in self.vertex and 4 in self.vertex):
                temp_reward = 7
            elif (6 in self.vertex):
                temp_reward = 6
            else:
                temp_reward = self.vertex[0]
            r2 = - 1.0 * abs(self.state.get_value(temp_reward) - self.state.get_last_value(temp_reward))
        # print("score_center_plus",score_center_plus,"r2:",r2)
        return self.reward_scale[0] * score_center_plus + self.reward_scale[1] * r2

    def step(self, action):
        """
        :param action: 注意是 list 格式，只有1个维度，表示干预node为何值
        :return:
        """
        info = dict()
        self.state.intervene(action)  # action是一个列表, 设置policy给出x4的值
        observed_vals, now_insulin = self.state.sample_all()  # 干预完了后获得可观测node的value
        r = self.reward(now_insulin, observed_vals).numpy()  # 使得 insulin 在 70 到 180 范围内
        # print("Step:", self.state.info_steps, "Action:", action, "Now_insulin:", now_insulin, "Reward:", r, 'state:', observed_vals)

        if self.state.info_steps == self.state.info_phase_length:  # 最后一步
            self.ep_rew = self.ep_rew + r  # 计算累计奖励
            info["episode"] = dict()
            info["episode"]["r"] = self.ep_rew  # episode的reward
            info["episode"]["l"] = self.state.info_steps  # episode的长度
            self.ep_rew = 0.0
            done = True
        else:
            self.ep_rew = self.ep_rew + r  # 计算累计奖励
            done = False

        # 每个step都更新一下，重新采样goal
        if len(self.list_last_vertex) > 0:
            goal_temp = []
            for idx in range(len(self.reward_goal) - 1, -1, -1):  # 逆序遍历
                if idx == len(self.reward_goal) - 1:  # x3-x12
                    action_temp, _states = self.reward_goal[idx].predict(self.reward_state[idx])
                    # print("self.reward_state[idx]:", self.reward_state[idx], "action_temp:", action_temp)
                    obs_temp, _rewards, _dones, _info = self.reward_env[idx].envs[0].step(action_temp)
                    self.reward_state[idx] = copy.deepcopy(obs_temp)
                    goal_temp = []
                    for item in self.reward_env[idx].envs[0].vertex:  # 存储goal，下个阶段的model需要使用
                        goal_temp.append([self.reward_env[idx].envs[0].state.get_value(item) - 10.0,
                                          self.reward_env[idx].envs[0].state.get_value(item) + 10.0])
                else:
                    # 先修改当前goal
                    self.reward_env[idx].envs[0].goal = copy.deepcopy(goal_temp)
                    action_temp, _states = self.reward_goal[idx].predict(self.reward_state[idx])
                    obs_temp, _rewards, _dones, _info = self.reward_env[idx].envs[0].step(action_temp)
                    self.reward_state[idx] = copy.deepcopy(obs_temp)
                    goal_temp = []
                    for item in self.reward_env[idx].envs[0].vertex:  # 存储goal，下个阶段的model需要使用
                        goal_temp.append([self.reward_env[idx].envs[0].state.get_value(item) - 10.0,
                                          self.reward_env[idx].envs[0].state.get_value(item) + 10.0])
            self.goal = copy.deepcopy(goal_temp)
        # print("self.goal:", self.goal)
        # concatenate all data that goes into an observation
        obs_tuple = np.hstack([observed_vals, reduce(operator.add, self.goal)])
        ####################################################### 这里考虑 1）observed_vals； 2）observed_vals, self.state.prev_action, self.state.prev_reward，再加上上一时刻状态
        obs = obs_tuple
        new_prev_action = np.array(action)
        self.state.step_state(new_prev_action, np.array([r]), obs)
        return obs, r, done, info

    def log_callback(self):
        for k, v in self.log_data.items():
            self.logger.logkv(k, v)
        self.log_data = defaultdict(int)

    def reset(self):
        self.state.reset()
        observed_vals, now_insulin = self.state.sample_all()
        obs_tuple = np.hstack((observed_vals, reduce(operator.add, self.goal)))
        obs = obs_tuple
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=94566):
        np.random.seed(seed)


class EnvState(object):
    def __init__(self, vertex, last_vertex, info_phase_length=50):
        """Create an object which holds the state of a CBNEnv"""
        self.info_phase_length = info_phase_length
        self.info_steps = None  # 用来记录当前是第几步
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.graph = None
        self.vertex = vertex
        self.last_vertex = last_vertex

        self.reset()

    def step_state(self, new_prev_action, new_prev_reward, new_prev_state):
        self.prev_action = new_prev_action
        self.prev_reward = new_prev_reward
        self.prev_state = copy.deepcopy(new_prev_state)
        if self.info_steps == self.info_phase_length:  # 最后一步
            self.info_steps = 0
        else:
            self.info_steps += 1

    def intervene(self, intervene_val):  # 干预
        self.graph.intervene(intervene_val)

    def sample_all(self):
        # return self.graph.sample_all()[:-1]   # 这里应该是设置最后一个node不可见
        return self.graph.sample_all()

    def get_value(self, node_idx):
        return self.graph.get_value(node_idx)

    def get_last_value(self, node_idx):
        return self.graph.get_last_value(node_idx)


    def get_graph(self):
        raise NotImplementedError()

    def reset(self):
        self.info_steps = 0
        self.prev_action = np.zeros(1)
        self.prev_reward = np.zeros(1)
        self.graph = self.get_graph()
        self.prev_state, _ = self.sample_all()


class TrainEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=True, vertex=self.vertex, last_vertex=self.last_vertex)


class TestEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=False, vertex=self.vertex, last_vertex=self.last_vertex)


class DebugEnvState(EnvState):
    def __init__(self):
        super().__init__()
        self.reward_data = None

