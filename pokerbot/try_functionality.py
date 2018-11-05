
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def run(nb_episodes=15):

    # parameters of environment
    starting_stack = 100
    big_blind = 10
    max_nb_hands = 10
    is_fixed_limit = True
    batch_size = 32

    # create agent
    agent = DQNAgent(starting_stack, "Q-Lee")

    # create opponents
    # player_one = HumanPlayer(starting_stack, "Sapiens")
    # player_two = RandomPlayer(starting_stack, "Hazard")
    # player_three = FishPlayer(starting_stack, "Nemo")
    # player_four = RandomPlayer(starting_stack, "Random")
    # player_five = StartingHandPlayer(starting_stack, "Tight")
    # player_six = StrengthHandPlayer(starting_stack, "Carlo")

    # create the environment
    env = HuGame(max_nb_hands, big_blind, agent, player_five, is_fixed_limit)

    # total reward for episode
    total_reward = 0

    # performance metrics - one per episode
    total_reward_list = []
    nb_hands_list = []
    epsilon_list = []
    action_type_list_of_lists = []

    # training
    for e in range(nb_episodes):

        # analytical results - many per episode
        action_type_list = []

        # catching bugs
        if agent.stack + player_five.stack != 2 * starting_stack:
            break
        if total_reward > starting_stack:
            break

        # reset environment
        env.reset()
        # and get initial state as row vector
        state, hand_over = env.initial_step()
        # if hand is over, need to do initial step again
        if hand_over:
            # discard all hands where opponent fold right away
            # TODO: is it the right thing to do?
            while hand_over:
                state, hand_over = env.initial_step()
        # reshape state into something ANN understands
        state = np.reshape(np.array(state), [1, 15])

        while not env.game_over:

            # get action from agent following an epsilon greedy policy
            action = agent.act(state)
            logging.debug("agent action: {}".format(action))
            # analytical results - many per episode
            action_type_list.append(action)

            # translate numerical action into str action
            possible_actions = env.current_hand.possible_actions
            logging.debug("possible actions: {}".format(possible_actions))
            if action == 0:
                if 'check' in possible_actions:
                    str_action = 'check'
                elif 'call' in possible_actions:
                    str_action = 'call'
                else:
                    str_action = 'all-in'
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
                if 'check' in possible_actions:
                    str_action = 'check'
                else:
                    str_action = 'fold'
            logging.debug("action applied: {}".format(str_action))

            # apply action into environment and observe feedback
            next_state, reward, game_done, hand_over, info = \
                env.step(str_action)

            # reshape state into something ANN understands
            next_state = np.reshape(np.array(next_state), [1, 15])

            # add sequence to memory
            agent.remember(state, action, reward, next_state, hand_over)

            # if done print relevant metrics and exit episode
            if game_done:
                total_reward = env.player_hero.stack - starting_stack
                print("episode: {}/{}, nb_hands: {}, e: {:.2}, reward: {}"
                      .format(e + 1, nb_episodes, env.hand_number,
                              agent.epsilon, total_reward))

                # performance metrics - one per episode
                total_reward_list.append(total_reward)
                nb_hands_list.append(env.hand_number)
                epsilon_list.append(agent.epsilon)
                action_type_list_of_lists.append(action_type_list)
                break

            # reinitialize variables
            # if hand is over, need to do initial step again
            if hand_over:
                # discard all hands where opponent fold right away
                # TODO: is it the right thing to do?
                while hand_over:
                    state, hand_over = env.initial_step()
                # reshape state into something ANN understands
                state = np.reshape(np.array(state), [1, 15])
            else:
                state = next_state

            # if memory is big enough, replay a batch of examples
            # and learn from them
            # this piece is also responsible for the decay in epsilon
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    # concatenate results
    results = [total_reward_list, nb_hands_list,
               epsilon_list, action_type_list_of_lists]

    return agent, env, results


def visualize_results(results, env):
    # put results in a DataFrame
    df = pd.DataFrame({'rewards': results[0], 'nb_hands': results[1],
                       'epsilon': results[2], 'action_type': results[3]})
    # adding features from action type lists
    df['count_0'] = df['action_type'].apply(lambda x: x.count(0))
    df['count_1'] = df['action_type'].apply(lambda x: x.count(1))
    df['count_2'] = df['action_type'].apply(lambda x: x.count(2))
    df['count_0_rw100'] = df['count_0'].rolling(window=100).mean()
    df['count_1_rw100'] = df['count_1'].rolling(window=100).mean()
    df['count_2_rw100'] = df['count_2'].rolling(window=100).mean()
    df['total_rw100'] = df['count_0_rw100'] + df['count_1_rw100'] + df[
        'count_2_rw100']
    # close all open figures
    plt.close("all")
    # create subplots
    fig, ax = plt.subplots(2, 2)
    # title with import number of parameters
    fig.suptitle("Training versus {} \nEpisodes: {} - Hands {} - Actions {}"
                 .format(type(env.player_villain).__name__,
                         len(results[0]),
                         sum(results[1]),
                         sum([sum(action_list)
                              for action_list in results[3]])),
                 fontsize=14)
    # rewards on a rolling window
    ax[0, 0].plot(df['rewards'].rolling(window=50).mean(), label='50')
    ax[0, 0].plot(df['rewards'].rolling(window=100).mean(), label='100')
    ax[0, 0].set_title("Rewards per episode on rolling window")
    ax[0, 0].set_ylabel("reward")
    ax[0, 0].legend(loc='upper left')
    # number of hands on a rolling window
    ax[0, 1].plot(df['nb_hands'].rolling(window=50).mean(), label='50')
    ax[0, 1].plot(df['nb_hands'].rolling(window=100).mean(), label='100')
    ax[0, 1].set_title("Number of hands per episode on rolling window")
    ax[0, 1].set_ylabel("number of hands")
    ax[0, 1].legend(loc='upper left')
    # epsilon
    ax[1, 0].plot(df['epsilon'])
    ax[1, 0].set_title("Epsilon")
    ax[1, 0].set_ylabel("epsilon")
    ax[1, 0].set_xlabel("episodes")
    # action types on a rolling window
    ax[1, 1].stackplot(range(1, len(df['count_0_rw100']) + 1),
                       df["count_0_rw100"] / df['total_rw100'],
                       df["count_1_rw100"] / df['total_rw100'],
                       df["count_2_rw100"] / df['total_rw100'],
                       labels=['0', '1', '2'])
    ax[1, 1].set_title("Action type breakdown")
    ax[1, 1].set_ylabel("proportion (%)")
    ax[1, 1].set_xlabel("episodes")
    ax[1, 1].legend(loc='upper left')
    return fig, ax

