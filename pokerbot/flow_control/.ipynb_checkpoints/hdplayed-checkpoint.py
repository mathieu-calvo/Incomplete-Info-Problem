
import logging
from itertools import cycle

from ..flow_control.deck import Deck
from ..hand_evaluation.hand import Hand, compare_two_hands
from ..agent.dqnagent import DQNAgent, DRQNAgent
from ..opponents.humanplayer import HumanPlayer
from ..globals import SEQUENCE_ACTIONS_ID

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


def create_stage_generator():
    """
    Create the sequence of stages the hand can go through
    """
    stage_sequence = ['pre-flop', 'flop', 'turn', 'river', 'showdown']
    for i in stage_sequence:
        yield i


class HdPlayed(object):
    """
    Hand being played object managing flow control for the hand played
    between two poker players, based on the current parameters of the game.
    Keeps track of what is happening during the hand as well.

    Attributes:
        # parameters
        hero_is_big_blind (bool): position of the hero
        playerBB (class.Player): player who will play big blind on this hand
        playerSB (class.Player): player who will play small blind on this hand
        is_fixed_limit (bool): fixed limit game if True, no-limit if False
        big_blind (int): initial compulsory stake
        small_blind (int): second initial compulsory stake
        handBB (class.Hand): hand object of player on big blind
        handSB (class.Hand): hand object of player on small blind
        flop (list): list of communal cards coming on the flop
        turn (list): list containing communal card coming on the turn
        river (list): list containing communal card coming on the river
        hand_number (int): index to keep track of number of the hand played
        # variables
        pot_size (int): size of the pot on that hand
        hand_over (bool): indicating whether hand is over
        # hand history objects
        hand_history_BB (str): hand history object seen from BB player
        hand_history_SB (str): hand history object seen from SB player
        json_hand_hist_BB (dict): hand history in json format
        json_hand_hist_SB (dict): hand history in json format
        state_SB (array): state of the environment as seen from SB player
        state_BB (array): state of the environment as seen from BB player
    """

    def __init__(self, hero_is_big_blind, player_hero, player_villain,
                 big_blind, is_fixed_limit, cards, hand_number):
        """
        Instantiate a hand played object based on players, parameters,
        and list of 9 randomly drawn cards
        e.g. HdPlayed(True, RandomPlayer(100,'Joe'), RandomPlayer(100,
        'Mike'), 10, True, Deck().deal_cards(9), 3)
        """
        # parameters
        self.hero_is_big_blind = hero_is_big_blind
        self.player_hero = player_hero
        self.player_villain = player_villain
        if self.hero_is_big_blind:
            self.playerBB = self.player_hero
            self.playerSB = self.player_villain
        else:
            self.playerBB = self.player_villain
            self.playerSB = self.player_hero
        self.big_blind = big_blind
        self.is_fixed_limit = is_fixed_limit
        self.hand_nb = hand_number
        self.small_blind = int(big_blind / 2)
        self.handBB = Hand([cards[0], cards[1]])
        self.handSB = Hand([cards[2], cards[3]])
        self.flop = [cards[4], cards[5], cards[6]]
        self.turn = [cards[7]]
        self.river = [cards[8]]
        # variables
        self.stage_sequence = create_stage_generator()
        self.pot_size = 0
        self.hero_reward = 0
        self.hand_over = False
        self.someone_has_folded = False
        self.someone_is_all_in = False
        self.stage = ""
        # betting round variables
        self.imbalance_size = self.big_blind - self.small_blind
        self.action_cycle = cycle([self.playerSB, self.playerBB])
        self.is_action_on_bb_cycle = cycle([False, True])
        self.nb_actions = 0
        self.active_player = self.action_cycle.__next__()
        self.is_action_on_bb = self.is_action_on_bb_cycle.__next__()
        self.action_trail = ''
        self.possible_actions = self._get_possible_actions()
        # hand history objects
        self.hand_history_BB, self.json_hand_hist_BB, self.state_BB = \
            self._initialize_hand_history(self.playerBB)
        self.hand_history_SB, self.json_hand_hist_SB, self.state_SB = \
            self._initialize_hand_history(self.playerSB)
        # active hand history
        self.active_json_hist = self._get_active_json_hist()

    def _next_stage(self):
        """
        Iterate through the sequence of stages the hand can go through
        Update relevant attributes for the betting round accordingly and
        log important information
        """
        self.stage = self.stage_sequence.__next__()

        if self.stage != 'pre-flop':
            self.action_cycle = cycle([self.playerBB, self.playerSB])
            self.is_action_on_bb_cycle = cycle([True, False])
            self.imbalance_size = 0
            self.nb_actions = 0
            self.active_player = self.action_cycle.__next__()
            self.is_action_on_bb = self.is_action_on_bb_cycle.__next__()
            self.action_trail = ''

        if self.stage == "pre-flop":
            # blinds
            logging.debug('{} is Big Blind'.format(self.playerBB.name))
            # check if one of the players is all in under the blind
            if self.playerSB.stack <= self.small_blind:
                self.someone_is_all_in = True
                all_in_under_sb = self.playerSB.stack
                self.playerSB.bet_amount(all_in_under_sb)
                # in the heads-up case, BB has necessarily more
                self.playerBB.bet_amount(all_in_under_sb)
                self.playerSB.bet_amount(all_in_under_sb)
                self.pot_size += all_in_under_sb * 2
                self.imbalance_size = 0
            if self.playerBB.stack <= self.big_blind:
                self.someone_is_all_in = True
                all_in_under_bb = self.playerBB.stack
                self.playerBB.bet_amount(all_in_under_bb)
                # in the heads-up case, SB has necessarily more
                # but SB could still have a choice to make
                if all_in_under_bb <= self.small_blind:
                    self.playerBB.bet_amount(all_in_under_bb)
                    self.playerSB.bet_amount(all_in_under_bb)
                    self.pot_size += all_in_under_bb * 2
                    self.imbalance_size = 0
                else:
                    self.playerBB.bet_amount(all_in_under_bb)
                    self.playerSB.bet_amount(self.small_blind)
                    self.pot_size += self.small_blind + all_in_under_bb
                    self.imbalance_size = all_in_under_bb - self.small_blind
            # normal case - everyone can afford the initial stakes
            if not self.someone_is_all_in:
                self.playerBB.bet_amount(self.big_blind)
                self.playerSB.bet_amount(self.small_blind)
                self.pot_size += self.big_blind + self.small_blind
                self.imbalance_size = self.big_blind - self.small_blind
            # log players' private cards
            logging.debug('{} has {}'.format(self.playerBB.name,
                                             self.handBB.private_cards))
            logging.debug('{} has {}'.format(self.playerSB.name,
                                             self.handSB.private_cards))

        elif self.stage == "flop":
            # log cards
            logging.debug('Flop comes {}'.format(self.flop))
            self.json_hand_hist_BB['community_cards'] += self.flop
            self.json_hand_hist_SB['community_cards'] += self.flop
            self.update_hand_histories("***** Dealing flop: {}\n"
                                       .format(self.flop))
            # log states - for lack of a better idea for now...
            self.state_BB[5] = self.flop[0].numerical_id
            self.state_BB[6] = self.flop[1].numerical_id
            self.state_BB[7] = self.flop[2].numerical_id
            self.state_SB[5] = self.flop[0].numerical_id
            self.state_SB[6] = self.flop[1].numerical_id
            self.state_SB[7] = self.flop[2].numerical_id
            # integrate info into hands
            self.handBB.add_public_cards(self.flop)
            self.handSB.add_public_cards(self.flop)

        elif self.stage == "turn":
            # log cards
            logging.debug('Turn comes {}'.format(self.turn))
            self.json_hand_hist_BB['community_cards'] += self.turn
            self.json_hand_hist_SB['community_cards'] += self.turn
            self.update_hand_histories("***** Dealing turn: {} - {}\n"
                                       .format(self.flop, self.turn))
            # log states - for lack of a better idea for now...
            self.state_SB[8] = self.turn[0].numerical_id
            self.state_BB[8] = self.turn[0].numerical_id
            # integrate info into hands
            self.handBB.add_public_cards(self.turn)
            self.handSB.add_public_cards(self.turn)

        elif self.stage == "river":
            # log cards
            logging.debug('River comes {}'.format(self.river))
            self.json_hand_hist_BB['community_cards'] += self.river
            self.json_hand_hist_SB['community_cards'] += self.river
            self.update_hand_histories("***** Dealing river: {} - {} - {}\n"
                                       .format(self.flop, self.turn,
                                               self.river))
            # log states - for lack of a better idea for now...
            self.state_SB[9] = self.river[0].numerical_id
            self.state_BB[9] = self.river[0].numerical_id
            # integrate info into hands
            self.handBB.add_public_cards(self.river)
            self.handSB.add_public_cards(self.river)

        elif self.stage == "showdown":
            self.hand_over = True
            # Evaluate best combinations
            self.handBB.update_best_combination()
            self.handSB.update_best_combination()
            # Evaluate winner at showdown
            self.update_hand_histories("***** Summary: \n"
                                       "{} shows {}, best hand is {}, {}\n"
                                       "{} shows {}, best hand is {}, {}\n"
                                       "Pot size: {}\n"
                                       .format(self.playerBB.name,
                                               self.handBB.private_cards,
                                               self.handBB.best_combination,
                                               self.handBB
                                               .human_readable_rank(),
                                               self.playerSB.name,
                                               self.handSB.private_cards,
                                               self.handSB.best_combination,
                                               self.handSB
                                               .human_readable_rank(),
                                               self.pot_size))
            # evaluate winner at showdown
            winner = compare_two_hands(self.handBB, self.handSB)
            if winner == 'hand1':
                self.playerBB.win_pot(self.pot_size)
                self.update_hand_histories('{} wins the pot: +{}$'
                                           .format(self.playerBB.name,
                                                   self.pot_size))
                if self.hero_is_big_blind:
                    self.hero_reward = \
                        (self.pot_size - self.imbalance_size) / 2
                else:
                    self.hero_reward = \
                        -(self.pot_size - self.imbalance_size) / 2

            elif winner == 'hand2':
                self.playerSB.win_pot(self.pot_size)
                self.update_hand_histories('{} wins the pot: +{}$'
                                           .format(self.playerSB.name,
                                                   self.pot_size))
                if self.hero_is_big_blind:
                    self.hero_reward = \
                        -(self.pot_size - self.imbalance_size) / 2
                else:
                    self.hero_reward = \
                        (self.pot_size - self.imbalance_size) / 2

            else:
                self.playerBB.split_pot(self.pot_size)
                self.playerSB.split_pot(self.pot_size)
                self.update_hand_histories("Splitting the pot")
                self.hero_reward = 0

    def _get_hero_hand_history(self):
        """
        Get the human-readable hand history from the hero point of view
        """
        if self.hero_is_big_blind:
            return self.hand_history_BB
        else:
            return self.hand_history_SB

    def _get_hero_state(self):
        """
        Get the state of the environment from the hero point of view
        """
        if self.hero_is_big_blind:
            return self.state_BB
        else:
            return self.state_SB

    def _get_possible_actions(self):
        """
        Getting possible actions at any moment of the betting round, based on
        context, and player whose turn it is
        """
        if self.imbalance_size > 0:
            if self.imbalance_size >= self.active_player.stack:
                return ['all-in', 'fold']
            else:
                if self.someone_is_all_in:
                    return ['call', 'fold']
                else:
                    if self.is_fixed_limit:
                        # if number of actions above threshold, bet is capped
                        if self.nb_actions >= 4:
                            return ['call', 'fold']
                    return ['call', 'raise', 'fold']
        else:
            return ['check', 'bet']

    def _update_possible_actions(self):
        """
        Update set of possible actions for active player at any given moment
        """
        self.possible_actions = self._get_possible_actions()

    def _get_active_json_hist(self):
        """
        Get json hand history from the point of view of the active player
        """
        if self.is_action_on_bb:
            return self.json_hand_hist_BB
        return self.json_hand_hist_SB

    def _enforce_action(self, action):
        """
        Update attributes based on action taken by player
        Also, calls the function to update the state accordingly
        """
        if action == 'fold':
            self.update_hand_histories("{} folds\n"
                                       .format(self.active_player.name))
            self.someone_has_folded = True
            self.hand_over = True
            # attribute winnings to other player
            other_player = self.action_cycle.__next__()
            other_player.win_pot(self.pot_size)
            self.update_hand_histories('{} wins the pot: {}$'
                                       .format(other_player.name,
                                               self.pot_size))
            # update reward
            if self.hero_is_big_blind:
                if self.is_action_on_bb:
                    self.hero_reward = \
                        - (self.pot_size - self.imbalance_size) / 2
                else:
                    self.hero_reward = \
                        (self.pot_size - self.imbalance_size) / 2
            else:
                if self.is_action_on_bb:
                    self.hero_reward = \
                        (self.pot_size - self.imbalance_size) / 2
                else:
                    self.hero_reward = \
                        -(self.pot_size - self.imbalance_size) / 2

        elif action == 'check':
            self.update_hand_histories("{} checks\n"
                                       .format(self.active_player.name))

        elif action == 'call':
            self.update_hand_histories("{} calls\n"
                                       .format(self.active_player.name))
            self.active_player.bet_amount(self.imbalance_size)
            self.pot_size += self.imbalance_size
            self.imbalance_size = 0

        elif action == 'bet':
            if self.is_fixed_limit:
                if self.stage in ['turn', 'river']:
                    bet_size = min(self.big_blind * 2,
                                   self.active_player.stack)
                else:
                    bet_size = min(self.big_blind, self.active_player.stack)
            else:
                bet_size = self.active_player.choose_amount(
                    minimum=self.big_blind,
                    maximum=self.active_player.stack,
                    pot_size=self.pot_size)
            self.active_player.bet_amount(bet_size)
            self.pot_size += bet_size
            if self.active_player.stack == 0:  # check if player is all in
                self.update_hand_histories("{} bets {}, and is all-in\n"
                                           .format(self.active_player.name,
                                                   bet_size))
                self.someone_is_all_in = True
            else:
                self.update_hand_histories("{} bets {}\n"
                                           .format(self.active_player.name,
                                                   bet_size))
            self.imbalance_size = bet_size

        elif action == 'raise':
            # minimum raise is calling the imbalance and doubling it
            # except pre-flop where the imbalance is the sb but the raise
            # needs to be at least the bb - or if all in
            min_raise = self.imbalance_size + min(max(self.imbalance_size,
                                                      self.big_blind),
                                                  self.active_player.stack -
                                                  self.imbalance_size)
            if self.is_fixed_limit:
                raise_size = min_raise
            else:
                raise_size = self.active_player.choose_amount(
                    minimum=min_raise,
                    maximum=self.active_player.stack,
                    pot_size=self.pot_size)
            self.active_player.bet_amount(raise_size)
            self.pot_size += raise_size
            amount_on_top = raise_size - self.imbalance_size
            if self.active_player.stack == 0:  # check if player is all in
                if amount_on_top > 0:
                    self.update_hand_histories("{} raises {}, and is all-in\n"
                                               .format(self.active_player.name,
                                                       amount_on_top))
                else:
                    self.update_hand_histories("{} calls {}, and is all-in\n"
                                               .format(self.active_player.name,
                                                       raise_size))
                self.someone_is_all_in = True
            else:
                self.update_hand_histories("{} raises {}\n"
                                           .format(self.active_player.name,
                                                   amount_on_top))
            self.imbalance_size = amount_on_top
            # if someone has gone all-in, there may be an imbalance left and
            # the first player to have moved may have had more chips,
            # in that case he needs to get back from the pot
            if self.imbalance_size < 0:
                # player will always be the first mover
                # they can't be a negative imbalance if the person who
                # concludes the betting round has more chips
                self.action_cycle.__next__() \
                    .get_back_from_pot(-self.imbalance_size)
                self.pot_size += self.imbalance_size
                self.imbalance_size = 0

        elif action == 'all-in':
            if self.someone_is_all_in:
                all_in_amount = min(self.active_player.stack,
                                    self.imbalance_size)
            else:
                all_in_amount = self.active_player.stack
                self.someone_is_all_in = True
            self.active_player.bet_amount(all_in_amount)
            self.pot_size += all_in_amount
            amount_on_top = all_in_amount - self.imbalance_size
            if amount_on_top > 0:
                self.update_hand_histories("{} bets {}, and is all-in\n"
                                           .format(self.active_player.name,
                                                   amount_on_top))
            else:
                self.update_hand_histories("{} calls {}, and is all-in\n"
                                           .format(self.active_player.name,
                                                   all_in_amount))
            self.imbalance_size = amount_on_top
            # if someone has gone all-in, there may be an imbalance left and
            # the first player to have moved may have had more chips,
            # in that case he needs to get back from the pot
            if self.imbalance_size < 0:
                # player will always be the first mover
                # they can't be a negative imbalance if the person who
                # concludes the betting round has more chips
                self.action_cycle.__next__()\
                    .get_back_from_pot(-self.imbalance_size)
                self.pot_size += self.imbalance_size
                self.imbalance_size = 0

        # increment action count
        self.nb_actions += 1
        # update state accordingly
        self._update_state(action)

    def _next_turn(self):
        """
        Update attributes in order to change turn
        """
        self.active_player = self.action_cycle.__next__()
        self.is_action_on_bb = self.is_action_on_bb_cycle.__next__()

    def _is_betting_round_over(self):
        """
        Check whether conditions for end of betting round have been met
        """
        if (self.nb_actions < 2) or \
                (self.nb_actions >= 2 and self.imbalance_size > 0):
            return False
        return True

    def _update_state(self, action):
        """
        Update state based on action taken by player
        """

        if action in ['call', 'check', 'all-in']:
            self.action_trail += 'C'
        elif action in ['bet', 'raise']:
            self.action_trail += 'B'
        elif action == 'fold':
            self.action_trail += 'F'

        # update stacks and pot size
        self.state_BB[0] = self.playerBB.stack
        self.state_BB[1] = self.playerSB.stack
        self.state_SB[0] = self.playerSB.stack
        self.state_SB[1] = self.playerBB.stack
        self.state_BB[-1] = self.pot_size
        self.state_SB[-1] = self.pot_size

        # need to update action trail for the stage
        if self.stage == 'pre-flop':
            self.state_BB[-5] = SEQUENCE_ACTIONS_ID[self.action_trail]
            self.state_SB[-5] = SEQUENCE_ACTIONS_ID[self.action_trail]
        elif self.stage == 'flop':
            self.state_BB[-4] = SEQUENCE_ACTIONS_ID[self.action_trail]
            self.state_SB[-4] = SEQUENCE_ACTIONS_ID[self.action_trail]
        elif self.stage == 'turn':
            self.state_BB[-3] = SEQUENCE_ACTIONS_ID[self.action_trail]
            self.state_SB[-3] = SEQUENCE_ACTIONS_ID[self.action_trail]
        elif self.stage == 'river':
            self.state_BB[-2] = SEQUENCE_ACTIONS_ID[self.action_trail]
            self.state_SB[-2] = SEQUENCE_ACTIONS_ID[self.action_trail]

    def initial_step(self):
        """
        Initialize the hand, with initial stakes, and beginning of the first
        betting round

        Returns:
            state (array): state of the environment
            hand_over (bool): indicating if opponent folds small blind
            info (object): hand history for debugging purposes
        """
        # checking it is called at the right moment
        self._next_stage()
        assert(self.stage == 'pre-flop')

        # let the opponent play if it is its turn to - can be twice in a row
        while self.active_player == self.player_villain:
            
            # update possible actions
            self._update_possible_actions()
        
            # show information if user is human
            if isinstance(self.active_player, HumanPlayer):
                if self.is_action_on_bb:
                    logging.info("{}".format(self.hand_history_BB))
                else:
                    logging.info("{}".format(self.hand_history_SB))
                    
            # getting action from player
            action = self.active_player.take_action(self.possible_actions,
                                                    hand_hist=self.
                                                    _get_active_json_hist())
            # enforce action
            self._enforce_action(action)

            if not self.hand_over:
                # change whose turn it is
                self._next_turn()

        # update possible actions
        self._update_possible_actions()
        # return initial state
        return self._get_hero_state(), self.hand_over, self.\
            _get_hero_hand_history()

    def step(self, action):
        """
        Method to take a step into the hand for the agent

        Args:
            action (str): action selected by agent

        Returns:
            next_state (array): numerical representation of environment
            reward (int): numerical reward
            hand_done (bool): indicating if hand is over
            info (object): hand history for debugging purposes
        """
        # make sure action is valid
        assert(action in self.possible_actions)

        # enforce action from agent
        self._enforce_action(action)

        # check if hand is over
        if self.hand_over:
            return self._get_hero_state(), self.hero_reward, \
                   self.hand_over, self._get_hero_hand_history()

        # if both all in, need to go to showdown
        if self.someone_is_all_in and self.imbalance_size == 0:
            logging.debug("Both all-in, going to showdown")
            while not self.hand_over:
                self._next_stage()
            return self._get_hero_state(), self.hero_reward, self\
                .hand_over, self._get_hero_hand_history()

        # check if betting round is over
        if self._is_betting_round_over():
            self._next_stage()
            logging.debug("Betting round is over, going to next stage, "
                          "on to {}".format(self.active_player.name))
            # if showdown, hand is over:
            if self.hand_over:
                return self._get_hero_state(), self.hero_reward, \
                       self.hand_over, self._get_hero_hand_history()
        else:  # if not action is on the opponent
            self._next_turn()
            logging.debug("Betting round continues, on to {}"
                          .format(self.active_player.name))

        # let the opponent play if it is its turn to - can be twice in a row
        while self.active_player == self.player_villain:
            
            # update possible actions
            self._update_possible_actions()
        
            # show information if user is human
            if isinstance(self.active_player, HumanPlayer):
                if self.is_action_on_bb:
                    logging.info("{}".format(self.hand_history_BB))
                else:
                    logging.info("{}".format(self.hand_history_SB))

            # getting action from player
            action = self.active_player.take_action(self.possible_actions,
                                                    hand_hist=self.
                                                    _get_active_json_hist())

            # enforce action
            self._enforce_action(action)

            # check if hand is over
            if self.hand_over:
                return self._get_hero_state(), self.hero_reward, \
                       self.hand_over, self._get_hero_hand_history()

            # if both all in, need to go to showdown
            if self.someone_is_all_in and self.imbalance_size == 0:
                logging.debug("Both all-in, going to showdown")
                while not self.hand_over:
                    self._next_stage()
                return self._get_hero_state(), self.hero_reward, self \
                    .hand_over, self._get_hero_hand_history()

            # check if betting round is over
            if self._is_betting_round_over():
                self._next_stage()
                logging.debug("Betting round is over, going to next stage, "
                              "on to {}".format(self.active_player.name))
                # if showdown, hand is over:
                if self.hand_over:
                    return self._get_hero_state(), self.hero_reward, \
                           self.hand_over, self._get_hero_hand_history()
            else:  # if not, action continues
                self._next_turn()
                logging.debug("Betting round continues, on to {}"
                              .format(self.active_player.name))

            # update possible actions
            self._update_possible_actions()

        # return new state and reward (if any) for the agent to see
        return self._get_hero_state(), self.hero_reward, self.hand_over, self\
            ._get_hero_hand_history()

    def reset(self):
        """
        Method to reset environment by dealing a new hand, resetting stacks and 
        changing positions
        """
        # update the hand count and change positions
        self.hand_nb += 1
        self.hero_is_big_blind = not self.hero_is_big_blind
        if self.hero_is_big_blind:
            self.playerBB = self.player_hero
            self.playerSB = self.player_villain
        else:
            self.playerBB = self.player_villain
            self.playerSB = self.player_hero
        # reset stacks
        self.playerSB.reset_stack()
        self.playerBB.reset_stack()
        # deal a new hand
        cards = Deck().deal_cards(9)
        self.handBB = Hand([cards[0], cards[1]])
        self.handSB = Hand([cards[2], cards[3]])
        self.flop = [cards[4], cards[5], cards[6]]
        self.turn = [cards[7]]
        self.river = [cards[8]]        
        # variables
        self.stage_sequence = create_stage_generator()
        self.pot_size = 0
        self.hero_reward = 0
        self.hand_over = False
        self.someone_has_folded = False
        self.someone_is_all_in = False
        self.stage = ""
        # betting round variables
        self.imbalance_size = self.big_blind - self.small_blind
        self.action_cycle = cycle([self.playerSB, self.playerBB])
        self.is_action_on_bb_cycle = cycle([False, True])
        self.nb_actions = 0
        self.active_player = self.action_cycle.__next__()
        self.is_action_on_bb = self.is_action_on_bb_cycle.__next__()
        self.action_trail = ''
        self.possible_actions = self._get_possible_actions()
        # hand history objects
        self.hand_history_BB, self.json_hand_hist_BB, self.state_BB = \
            self._initialize_hand_history(self.playerBB)
        self.hand_history_SB, self.json_hand_hist_SB, self.state_SB = \
            self._initialize_hand_history(self.playerSB)
        # active hand history
        self.active_json_hist = self._get_active_json_hist()
        
    def _initialize_hand_history(self, player):
        """
        Initialize hand history for a given player

        Args:
            player (subclass.Player): class object Player

        Returns:
            str_out (str): human readable hand history
            json_out (dict): machine readable hand history
            state_out (array): state of the environment in numerical format
        """
        # check what position the player is in
        if player == self.playerBB:
            stack = self.playerBB.stack
            opponent_stack = self.playerSB.stack
            position = "Big Blind"
            private_cards = self.handBB.private_cards
            simp_rep = self.handBB.get_simp_preflop_rep()
        else:
            stack = self.playerSB.stack
            opponent_stack = self.playerBB.stack
            position = "Small Blind"
            private_cards = self.handSB.private_cards
            simp_rep = self.handSB.get_simp_preflop_rep()

        if self.is_fixed_limit:
            game_type = 'Fixed Limit Texas Hold-em'
        else:
            game_type = "No Limit Texas Hold-em"

        str_out = "\n***** Hand #{} history\n".format(self.hand_nb) \
                  + "User: {}\n".format(player.name) \
                  + "Position: {}\n".format(position) \
                  + "Game type: {}\n".format(game_type) \
                  + "Stacks: {} ({}), {} ({})\n".format(self.playerBB.name,
                                                        self.playerBB.stack,
                                                        self.playerSB.name,
                                                        self.playerSB.stack) \
                  + "{} posts small blind({}) {} posts big " \
                    "blind({})\n".format(self.playerSB.name, self.small_blind,
                                         self.playerBB.name, self.big_blind) \
                  + "***** Dealing private cards " \
                    "to {}: {}\n".format(player.name, private_cards)

        json_out = {'position': {position},
                    'preflop': {'hole_cards': private_cards,
                                'simp_rep': simp_rep},
                    'community_cards': []}

        # STACKS - POSITION - CARDS, private&common - ACTION seq.- POT SIZE
        state_out = [
            stack,
            opponent_stack,
            0 if position == 'Small Blind' else 1,
            private_cards[0].numerical_id,
            private_cards[1].numerical_id,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            self.pot_size]

        return str_out, json_out, state_out

    def update_hand_histories(self, text):
        self.hand_history_BB += text
        self.hand_history_SB += text
