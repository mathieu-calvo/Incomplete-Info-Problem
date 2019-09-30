
import random
import numpy as np
import logging
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from ..flow_control.player import Player

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

    def __init__(self, stack, name,
                 epsilon_decay=0.995, learning_rate=0.01, gamma=0.95,
                 starting_epsilon=1.0, epsilon_min=0.01):
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
        self.gamma = gamma    # discount rate
        self.epsilon = starting_epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
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

    def loading_weights(self, file_path):
        self.model.load_weights(file_path)

    def loading_model(self, file_path):
        self.model = load_model(file_path)

    def saving_weights(self, file_path):
        self.model.save_weights(file_path)

    def saving_model(self, file_path):
        self.model.save(file_path)


class DRQNAgent(Player):
    """
    Poker player object capable of playing games
    Learning agent who improves its decision making by interacting with the
    environment and by training a Deep Recurrent Q Neural Network

    Inherits from the Player class
    Many methods allow the agent to learn by interacting with the environment
    """

    def __init__(self, stack, name,
                 epsilon_decay=0.995, learning_rate=0.01, gamma=0.95,
                 starting_epsilon=1.0, epsilon_min=0.01):
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
        self.memory_sequence = []
        self.gamma = gamma    # discount rate
        self.epsilon = starting_epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep Recurrent Q learning Model
        model = Sequential()
        model.add(LSTM(self.state_size,
                       batch_input_shape=(None, None, self.state_size),
                       return_sequences=True))
        model.add(LSTM(self.action_size,
                       return_sequences=False))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def add_to_memory_sequence(self, state):
        self.memory_sequence.append(state)

    def reset_memory_sequence(self):
        self.memory_sequence = []

    def remember_sequence(self, action, reward, next_state, hand_done):
        self.memory.append((self.memory_sequence,
                            action,
                            reward,
                            next_state,
                            hand_done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            logging.debug("agent acts randomly")
            return random.randrange(self.action_size)
        logging.debug("agent uses model to act")
        # expecting a three dimensional memory sequence
        mem_seq_array = np.array(self.memory_sequence)\
            .reshape([1, len(self.memory_sequence), self.state_size])
        act_values = self.model.predict(mem_seq_array)
        return np.argmax(act_values[0])  # returns action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size, mode_current_length=True):
        if mode_current_length:
            # select length of subset of sequences from triangular distribution
            # to give more probabilities to shorter sequences
            len_seq = len(self.memory_sequence)
        else:
            # select length of subset of sequences from triangular distribution
            # to give more probabilities to shorter sequences
            max_len = max([len(mem[0]) for mem in self.memory])
            len_seq = int(np.random.triangular(1, 3, max_len+1, 1))
        # select subset of sequences in memory with drawn length
        subset_memory = [mem for mem in self.memory if len(mem[0]) == len_seq]
        # randomly select a batch of sequence from this subset
        if len(subset_memory) == 0:
            return None  # exit function
        elif len(subset_memory) <= batch_size:
            mini_batch = subset_memory
        else:
            mini_batch = random.sample(subset_memory, batch_size)
        # replay examples and use them to train model
        for mem_seq, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                # making next memory sequence
                next_mem_seq = mem_seq + [next_state]
                # expecting a three dimensional memory sequence
                next_mem_seq_array = np.array(next_mem_seq) \
                    .reshape([1, len(next_mem_seq), self.state_size])
                # estimate reward based on model
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_mem_seq_array)))
            # expecting a three dimensional memory sequence
            mem_seq_array = np.array(mem_seq).reshape([1, len(mem_seq), self.state_size])
            # get prediction from model
            target_f = self.model.predict(mem_seq_array)
            # replace prediction for action with actual target
            target_f[0][action] = target
            # train the model
            self.model.fit(mem_seq_array, target_f, epochs=1, verbose=0)

    def loading_weights(self, file_path):
        self.model.load_weights(file_path)

    def loading_model(self, file_path):
        self.model = load_model(file_path)

    def saving_weights(self, file_path):
        self.model.save_weights(file_path)

    def saving_model(self, file_path):
        self.model.save(file_path)
