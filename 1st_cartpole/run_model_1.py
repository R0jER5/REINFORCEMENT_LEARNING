import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow

def build_model(statets, actions):
    model = tensorflow.keras.Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

from rl.agents import DQNAgent #there are different types of agents
from rl.policy import BoltzmannQPolicy #2 types of reinforcement learning, vlue based, policy based
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

env = gym.make('CartPole-v0')
actions= env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states,actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights('dqn_weights.h5f')
_ = dqn.test(env, nb_episodes= 5, visualize=True)