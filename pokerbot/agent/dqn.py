
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    # create env
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # create agent
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")

    # initial parameters
    done = False
    batch_size = 32
    EPISODES = 5

    # training
    for e in range(EPISODES):

        # reset environment and get initial state as row vector
        state = env.reset()
        state = np.reshape(state, [1, state_size])  # to row vector

        for time in range(500):

            # # visualize environment
            # env.render()

            # get action from agent following an epsilon greedy policy
            action = agent.act(state)

            # apply action into environment and observe feedback
            next_state, reward, done, _ = env.step(action)

            # reshape output
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            # add sequence to memory
            agent.remember(state, action, reward, next_state, done)

            # reinitialize variables
            state = next_state

            # if done print relevant metrics and exit
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e + 1, EPISODES, time, agent.epsilon))
                break

            # if memory is big enough, replay a batch of examples
            # and learn from them
            # this piece is also responsible for the decay in epsilon
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # # save every ten episodes
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
