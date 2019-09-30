
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from datetime import timedelta

from .dqnagent import DQNAgent, DRQNAgent
from ..flow_control.hugame import HuGame
from ..opponents.humanplayer import HumanPlayer
from ..opponents.randomplayer import RandomPlayer
from ..opponents.fixedpolicyplayer import StartingHandPlayer, \
    StrengthHandPlayer, FishPlayer

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)

# agent, env, results = run(nb_episodes=10, epsilon_decay=0.995)
# fig, ax = visualize_results(agent, env, results)

# nb_episodes=500
# starting_stack=1000
# big_blind=20
# max_nb_hands=50
# is_fixed_limit=True
# batch_size=25
# learning_rate=0.1
# gamma=0.8
# epsilon_decay=0.995
# starting_epsilon=1.0
# epsilon_min=0.0,
# loading_model=True
# opponent_cls=FishPlayer

# nb episodes - epsilon decay
# 1000 - 0.995
# 5000 - 0.999
# 10000 - 0.9995
# 50000 - 0.9999
# 100000 - 0.9995
# ...


def pickle_memory(agent):
    filename = 'agent_memory'
    outfile = open(filename, 'wb')
    pickle.dump(agent.memory, outfile)
    outfile.close()


def rescale_reward(reward_to_scale, starting_stack):
    """
    Rescale state by using min-max normalization

    Args: reward_to_scale (int)

    Returns: scaled_reward (float) - belongs to [-1, 1] interval
    """
    return reward_to_scale / starting_stack


def rescale_state(state_to_scale, starting_stack):
    """
    Rescale state by using min-max normalization

    Args: state_to_scale (array)

    Returns: scaled_state (list)
    """
    scaled_state = state_to_scale.copy()
    # stacks and pot size
    for i in [0, 1, -1]:
        scaled_state[i] = (state_to_scale[i] - 0) / (starting_stack * 2 - 0)
    # cards
    for k in range(3, 10):
        scaled_state[k] = (state_to_scale[k] - 0) / (52 - 0)
    # action sequences
    for j in range(10, 14):
        scaled_state[j] = (state_to_scale[j] - 0) / (24 - 0)
    return scaled_state


def run(nb_episodes=500, starting_stack=1000, big_blind=20,
        max_nb_hands=50, is_fixed_limit=True, batch_size=25,
        learning_rate=0.1, gamma=0.8, epsilon_decay=0.995,
        starting_epsilon=1.0, epsilon_min=0.0,
        loading_model=True, saving_model=True,
        agent_cls=DRQNAgent, opponent_cls=FishPlayer):

    # path for saving or loading models
    path = "./pokerbot/pokerbot/agent/models/"

    # create opponent
    opponent = opponent_cls(starting_stack, 'Villain')

    # create agent
    agent = agent_cls(starting_stack, "Hero",
                      learning_rate=learning_rate,
                      gamma=gamma,
                      epsilon_decay=epsilon_decay,
                      starting_epsilon=starting_epsilon,
                      epsilon_min=epsilon_min)

    # load existing knowledge
    if loading_model:
        agent.loading_model(path + "{}_vs_{}_model.h5"
                            .format(type(agent).__name__,
                                    type(opponent).__name__))

    # create the environment
    env = HuGame(max_nb_hands, big_blind, agent, opponent, is_fixed_limit)

    # total reward for episode
    total_reward = 0

    # performance metrics - one per episode
    total_reward_list = []
    nb_hands_list = []
    epsilon_list = []
    action_type_list_of_lists = []
    first_action_type_list_of_lists = []

    # time taking
    start_time = time.monotonic()

    # training
    for e in range(nb_episodes):

        # analytical results - many per episode
        action_type_list = []
        first_action_type_list = []
        waiting_for_first_action = True

        # catching bugs
        if env.hand_number > env.max_nb_hands:
            break
        if agent.stack + opponent.stack != 2 * starting_stack:
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
        # rescale state - pre-processing stage
        state = rescale_state(state, starting_stack)
        # reshape state into something ANN understands
        state = np.reshape(np.array(state), [15])

        while not env.game_over:

            # get action from agent following an epsilon greedy policy
            action = agent.act(state)
            logging.debug("agent action: {}".format(action))
            # analytical results - many per episode
            action_type_list.append(action)
            if waiting_for_first_action:
                first_action_type_list.append(action)
                waiting_for_first_action = False

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

            # rescale state and reward - pre-processing stage
            next_state = rescale_state(next_state, starting_stack)
            reward = rescale_reward(reward, starting_stack)

            # reshape state into something ANN understands
            next_state = np.reshape(np.array(next_state), [15])

            # add sequence to memory
            if agent_cls == DQNAgent:
                agent.remember(state, action, reward, next_state, hand_over)
            elif agent_cls == DRQNAgent:
                # add to the attribute list
                agent.add_to_memory_sequence(state)
                # remember the sequence
                agent.remember_sequence(action,
                                        reward,
                                        next_state,
                                        hand_over)

            # if done print relevant metrics and exit episode
            if game_done:
                total_reward = env.player_hero.stack - starting_stack
                print("episode: {}/{}, nb_hands: {}, e: {:.3}, reward: {}"
                      .format(e + 1, nb_episodes, env.hand_number,
                              agent.epsilon, total_reward))

                # performance metrics - one per episode
                total_reward_list.append(total_reward)
                nb_hands_list.append(env.hand_number)
                epsilon_list.append(agent.epsilon)
                action_type_list_of_lists.append(action_type_list)
                first_action_type_list_of_lists.append(first_action_type_list)
                break

            # reinitialize variables
            # if hand is over, need to do initial step again
            if hand_over:
                # discard all hands where opponent fold right away
                # TODO: is it the right thing to do?
                while hand_over and env.hand_number < env.max_nb_hands:
                    state, hand_over = env.initial_step()
                # check whether last hand was dealt and is over,
                # if so end episode
                if hand_over and env.hand_number >= env.max_nb_hands:

                    env.game_over = True

                    total_reward = env.player_hero.stack - starting_stack
                    print("episode: {}/{}, nb_hands: {}, e: {:.3}, reward: {}"
                          .format(e + 1, nb_episodes, env.hand_number,
                                  agent.epsilon, total_reward))

                    # performance metrics - one per episode
                    total_reward_list.append(total_reward)
                    nb_hands_list.append(env.hand_number)
                    epsilon_list.append(agent.epsilon)
                    action_type_list_of_lists.append(action_type_list)
                    first_action_type_list_of_lists.append(
                        first_action_type_list)

                    break
                # rescale state - pre-processing stage
                state = rescale_state(state, starting_stack)
                # reshape state into something ANN understands
                state = np.reshape(np.array(state), [15])
                # reinitialize variable
                waiting_for_first_action = True
                # recurrent agent specific tasks
                if agent_cls == DRQNAgent:
                    # reset memory sequence
                    agent.reset_memory_sequence()
                    # add state to the attribute list
                    agent.add_to_memory_sequence(state)
                # if memory is big enough, after each hand, replay
                # a batch of examples with current length of sequence and
                # learn from them
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size, mode_current_length=True)
            else:
                state = next_state

        # if memory is big enough, after each episode, replay a batch of
        # examples with a drawn length of sequence and learn from them
        # if len(agent.memory) > batch_size:
        #     agent.replay(batch_size, mode_current_length=False)

        # epsilon decay
        agent.decay_epsilon()

    # time
    end_time = time.monotonic()
    training_time = timedelta(seconds=end_time - start_time)

    # concatenate results
    results = [total_reward_list, nb_hands_list, epsilon_list,
               action_type_list_of_lists, first_action_type_list_of_lists,
               training_time]

    # save new knowledge
    if saving_model:
        agent.saving_model(path + "{}_vs_{}_model.h5"
                           .format(type(agent).__name__,
                                   type(opponent).__name__))

    return agent, env, results


def visualize_results(agent, env, results):
    # put results in a DataFrame
    df = pd.DataFrame({'rewards': results[0], 'nb_hands': results[1],
                       'epsilon': results[2], 'action_type': results[3],
                       'first_action_type': results[4]})
    # adding features from action type lists
    df['count_0'] = df['action_type'].apply(lambda x: x.count(0))
    df['count_1'] = df['action_type'].apply(lambda x: x.count(1))
    df['count_2'] = df['action_type'].apply(lambda x: x.count(2))
    df['count_0_rw100'] = df['count_0'].rolling(window=100).mean()
    df['count_1_rw100'] = df['count_1'].rolling(window=100).mean()
    df['count_2_rw100'] = df['count_2'].rolling(window=100).mean()
    df['total_rw100'] = df['count_0_rw100'] + df['count_1_rw100'] + df[
        'count_2_rw100']
    # adding features from first action type lists
    df['count_0_first'] = df['first_action_type'].apply(lambda x: x.count(0))
    df['count_1_first'] = df['first_action_type'].apply(lambda x: x.count(1))
    df['count_2_first'] = df['first_action_type'].apply(lambda x: x.count(2))
    df['count_0_first_rw100'] = df['count_0_first'].rolling(window=100).mean()
    df['count_1_first_rw100'] = df['count_1_first'].rolling(window=100).mean()
    df['count_2_first_rw100'] = df['count_2_first'].rolling(window=100).mean()
    df['total_first_rw100'] = df['count_0_first_rw100'] + \
                              df['count_1_first_rw100'] + \
                              df['count_2_first_rw100']
    # close all open figures
    plt.close("all")
    # create subplots
    fig, ax = plt.subplots(2, 2)
    # title with import number of parameters
    fig.suptitle("Training {} versus {} "
                 "\nLearning rate {} - Discount factor {} - Epsilon decay {} -"
                 " Starting stack {} - Big blind {} - Max nb hands {}"
                 "\nEpisodes: {} - Hands {} - Actions {} - Time {}"
                 .format(type(env.player_hero).__name__,
                         type(env.player_villain).__name__,
                         agent.learning_rate,
                         agent.gamma,
                         agent.epsilon_decay,
                         env.player_hero.initial_stack,
                         env.big_blind,
                         env.max_nb_hands,
                         len(results[0]),
                         sum(results[1]),
                         sum([len(action_list) for action_list in results[
                             3]]),
                         str(results[5])),
                 fontsize=14)
    # rewards on a rolling window
    ax[0, 0].plot(df['rewards'].rolling(window=50).mean(), label='50')
    ax[0, 0].plot(df['rewards'].rolling(window=100).mean(), label='100')
    ax[0, 0].set_title("Rewards per episode - rolling window")
    ax[0, 0].set_ylabel("reward")
    ax[0, 0].legend(loc='upper left')
    # adding epsilon
    ax2 = ax[0, 0].twinx()
    ax2.plot(df['epsilon'], color='r', linestyle='--')
    ax2.set_ylabel("epsilon")
    # number of hands on a rolling window
    ax[0, 1].plot(df['nb_hands'].rolling(window=50).mean(), label='50')
    ax[0, 1].plot(df['nb_hands'].rolling(window=100).mean(), label='100')
    ax[0, 1].set_title("Number of hands per episode - rolling window")
    ax[0, 1].set_ylabel("number of hands")
    ax[0, 1].legend(loc='upper left')
    # adding epsilon
    ax3 = ax[0, 1].twinx()
    ax3.plot(df['epsilon'], color='r', linestyle='--')
    ax3.set_ylabel("epsilon")
    # first action types on a rolling window
    ax[1, 0].stackplot(range(1, len(df['count_0_first_rw100']) + 1),
                       df["count_0_first_rw100"] / df['total_first_rw100'],
                       df["count_1_first_rw100"] / df['total_first_rw100'],
                       df["count_2_first_rw100"] / df['total_first_rw100'],
                       labels=['passive - call',
                               'aggressive',
                               'passive - fold'])
    ax[1, 0].set_title("First action type breakdown - rolling window 100")
    ax[1, 0].set_ylabel("proportion (%)")
    ax[1, 0].set_xlabel("episodes")
    ax[1, 0].legend(loc='upper left')
    # action types on a rolling window
    ax[1, 1].stackplot(range(1, len(df['count_0_rw100']) + 1),
                       df["count_0_rw100"] / df['total_rw100'],
                       df["count_1_rw100"] / df['total_rw100'],
                       df["count_2_rw100"] / df['total_rw100'],
                       labels=['passive - call',
                               'aggressive',
                               'passive - fold'])
    ax[1, 1].set_title("Action type breakdown - rolling window 100")
    ax[1, 1].set_ylabel("proportion (%)")
    ax[1, 1].set_xlabel("episodes")
    ax[1, 1].legend(loc='upper left')
    return fig, ax
