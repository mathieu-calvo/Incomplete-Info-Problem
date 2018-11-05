
import random
import numpy as np
import logging
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from pokerbot.flow_control.player import Player

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


class DQNAgent(Player):
    """
    Poker player object capable of playing games
    Learning agent who improves its decision making by interacting with the
    environment and by training a Deep Q Neural Network

    Inherits from the Player class
    Many methods allow the agent to learn by interacting with the environment
    """

    def __init__(self, stack, name):
        """
        Instantiating the object using a numeric stack and a name
        e.g. DQNAgent(100,"Joe")
        """
        self.initial_stack = stack
        self.stack = stack
        self.name = name
        self.state_size = 15  # dimension of row vector representing the env
        self.action_size = 3  # CALL (CHECK) - BET (RAISE) - FOLD
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
            logging.debug("agent acts randomly")
            return random.randrange(self.action_size)
        logging.debug("agent uses model to act")
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
