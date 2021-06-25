import gym
import random
####TEST RANDOM ENVIRONMENT WITH OPEN AI  GYM.
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()##it will make it move left or right
        action = random.choice([0,1])## It will make rendom choices 0 or 1 means left or right
        n_state, reward,done, info = env.step(action)##by this we get no of states our final reward...
        score+=reward
    print('Episode:{} Score:{}'.format(episode,score))

### BUILD DEEP LEARNING MODEL
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
model = build_model(states, actions)
model.summary()


###BUILD AGENT WITH KERAS-RL

from rl.agents import DQNAgent #there are different types of agents
from rl.policy import BoltzmannQPolicy #2 types of reinforcement learning, vlue based, policy based
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
##build it again if it gives error::
model = tensorflow.keras.Sequential()
model = build_model(states, actions)

##train our model
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


scores = dqn.test(env,nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

_ = dqn.test(env, nb_episodes=5, visualize=True)

dqn.save_weights('dqn_weights.h5f', overwrite=True)
