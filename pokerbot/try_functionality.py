
import logging
import numpy as np

from .agent.dqnagent import DQNAgent

# from .flow_control.headsupgame import HeadsUpGame
from .flow_control.hugame import HuGame

# from .opponents.humanplayer import HumanPlayer
# from .opponents.randomplayer import RandomPlayer
# from .opponents.fishplayer import FishPlayer
from .opponents.fixedpolicyplayer import StartingHandPlayer
# StrengthHandPlayer

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


def run():

    # parameters of environment
    starting_stack = 100
    big_blind = 10
    max_nb_hands = 10
    is_fixed_limit = True
    batch_size = 32
    nb_episodes = 5

    # create agent
    agent = DQNAgent(starting_stack, "Q-Lee")

    # create opponents
    # player_one = HumanPlayer(starting_stack, "Sapiens")
    # player_two = RandomPlayer(starting_stack, "Hazard")
    # player_three = FishPlayer(starting_stack, "Nemo")
    # player_four = RandomPlayer(starting_stack, "Random")
    player_five = StartingHandPlayer(starting_stack, "Tight")
    # player_six = StrengthHandPlayer(starting_stack, "Carlo")

    # create the environment
    env = HuGame(max_nb_hands, big_blind, agent, player_five, is_fixed_limit)

    # training
    for e in range(nb_episodes):

        # reset environment
        env.reset()
        # and get initial state as row vector
        state = env.initial_step()
        # reshape state into something ANN understands
        state = np.reshape(np.array(state), [1, 15])

        while not env.game_over:

            # get action from agent following an epsilon greedy policy
            action = agent.act(state)

            # translate numerical action into str action
            possible_actions = env.current_hand.possible_actions
            if action == 0:
                if 'check' in possible_actions:
                    str_action = 'check'
                elif 'call' in possible_actions:
                    str_action = 'call'
                else:
                    str_action = 'all*in'
            elif action == 1:
                if 'bet' in possible_actions:
                    str_action = 'bet'
                elif 'raise' in possible_actions:
                    str_action = 'raise'
                elif 'all-in' in possible_actions:
                    str_action = 'all-in'
                else:
                    str_action = 'call'
            else:
                str_action = 'fold'

            # apply action into environment and observe feedback
            next_state, reward, game_done, hand_over, info = \
                env.step(str_action)

            # reshape state into something ANN understands
            next_state = np.reshape(np.array(next_state), [1, 15])

            # add sequence to memory
            agent.remember(state, action, reward, next_state, hand_over)

            # reinitialize variables
            # if hand is over, need to do initial step again
            if hand_over:
                state = env.initial_step()
            else:
                state = next_state

            # if done print relevant metrics and exit episode
            if game_done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e + 1, nb_episodes, 0, agent.epsilon))
                break

            # if memory is big enough, replay a batch of examples
            # and learn from them
            # this piece is also responsible for the decay in epsilon
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
