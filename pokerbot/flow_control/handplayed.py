
import logging
from itertools import cycle

from ..hand_evaluation.hand import Hand, compare_two_hands
from ..opponents.humanplayer import HumanPlayer

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


class HandPlayed(object):
    """
    Hand being played object managing flow control for the hand played
    between two poker players, based on the current parameters of the game.
    Keeps track of what is happening during the hand as well.

    Attributes:
        playerBB (class.Player): player who will play big blind on this hand
        playerSB (class.Player): player who will play small blind on this hand
        pot_size (int): size of the pot on that hand
        big_blind (int): initial compulsory stake
        small_blind (int): second initial compulsory stake
        is_fixed_limit (bool): fixed limit game if True, no-limit if False
        handBB (class.Hand): hand object of player on big blind
        handSB (class.Hand): hand object of player on small blind
        flop (list): list of communal cards coming on the flop
        turn (list): list containing communal card coming on the turn
        river (list): list containing communal card coming on the river
        hand_history_BB (str): hand history object seen from BB player
        hand_history_SB (str): hand history object seen from SB player
        json_hand_hist_BB (dict): hand history in json format
        json_hand_hist_SB (dict): hand history in json format
        hand_number (int): index to keep track of number of the hand played
    """

    def initialize_hand_history(self, player):
        """
        Initialize hand history for a given player

        Args:
            player (subclass.Player): class object Player

        Returns:
            str_out (str): human readable hand history
            json_out (dict): machine readable hand history
        """
        # check what position the player is in
        if player == self.playerBB:
            position = "Big Blind"
            private_cards = self.handBB.private_cards
            simp_rep = self.handBB.get_simp_preflop_rep()
        else:
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

        return str_out, json_out

    def update_hand_histories(self, text):
        self.hand_history_BB += text
        self.hand_history_SB += text

    def __init__(self, player1, player2, big_blind,
                 is_fixed_limit, cards, hand_number):
        """
        Instantiate a hand played object based on players, parameters,
        and list of 9 randomly drawn cards
        e.g. HandPlayed(HumanPlayer(100,'Joe'), HumanPlayer(100,'Mike'),
                        10, False, Deck().deal_cards(9), 3)
        """
        self.playerBB = player1
        self.playerSB = player2
        self.big_blind = big_blind
        self.is_fixed_limit = is_fixed_limit
        self.hand_nb = hand_number
        self.pot_size = 0
        self.small_blind = int(big_blind / 2)
        self.handBB = Hand([cards[0], cards[1]])
        self.handSB = Hand([cards[2], cards[3]])
        self.flop = [cards[4], cards[5], cards[6]]
        self.turn = [cards[7]]
        self.river = [cards[8]]
        self.hand_history_BB = self.initialize_hand_history(self.playerBB)[0]
        self.hand_history_SB = self.initialize_hand_history(self.playerSB)[0]
        self.json_hand_hist_BB = self.initialize_hand_history(self.playerBB)[1]
        self.json_hand_hist_SB = self.initialize_hand_history(self.playerSB)[1]

    def get_possible_actions(self, player, imbalance_size,
                             other_player_is_all_in, nb_actions):
        """
        Getting possible actions at any moment of the betting round, based on
        context

        Args:
            player (class.Player): class object Player
            imbalance_size (int): pre action imbalance size, i.e.
            positive if one player has put more into the pot than the other
            other_player_is_all_in (bool): explicit, to set up maximum bet
            nb_actions (int): number of actions in betting round so far

        Returns:
            actions (list): set of str action the player can choose from
        """
        if imbalance_size > 0:
            if imbalance_size >= player.stack:
                return ['all-in', 'fold']
            else:
                if other_player_is_all_in:
                    return ['call', 'fold']
                else:
                    if self.is_fixed_limit:
                        # if number of actions above threshold, bet is capped
                        if nb_actions >= 4:
                            return ['call', 'fold']
                    return ['call', 'raise', 'fold']
        else:
            return ['check', 'bet']

    def get_action_from_player(self, player, actions, imbalance_size,
                               other_player_is_all_in, json_hand_hist):
        """
        Getting actions from players, method will vary depending on type of
        player (i.e. which class or subclass it belongs to)

        Args:
            player (class.Player): class object Player
            actions (list): set of str action the player can choose from
            imbalance_size (int): pre action imbalance size, i.e.
            positive if one player has put more into the pot than the other
            other_player_is_all_in (bool): explicit, to set up maximum bet
            json_hand_hist (dict): hand history in json format

        Returns:
            has_folded (bool): let know whether someone folded
            is_all_in (bool): let know whether someone is all in
            imbalance_size (int): post action imbalance size
        """

        # getting action from player
        choice = player.take_action(actions, hand_hist=json_hand_hist)

        # return meaningful parameters accordingly
        if choice == 'fold':
            self.update_hand_histories("{} folds\n".format(player.name))
            return True, False, imbalance_size
        elif choice == 'check':
            self.update_hand_histories("{} checks\n".format(player.name))
            return False, False, 0
        elif choice == 'call':
            self.update_hand_histories("{} calls\n".format(player.name))
            player.bet_amount(imbalance_size)
            self.pot_size += imbalance_size
            return False, False, 0
        elif choice == 'bet':
            if self.is_fixed_limit:
                bet_size = self.big_blind
            else:
                bet_size = player.choose_amount(minimum=self.big_blind,
                                                maximum=player.stack,
                                                pot_size=self.pot_size)
            player.bet_amount(bet_size)
            self.pot_size += bet_size
            # check if player is all in
            if player.stack == 0:
                self.update_hand_histories("{} bets {}, and is all-in\n"
                                           .format(player.name, bet_size))
                return False, True, bet_size
            self.update_hand_histories("{} bets {}\n".format(player.name,
                                                             bet_size))
            return False, False, bet_size
        elif choice == 'raise':
            # minimum raise is calling the imbalance and doubling it
            # except pre-flop where the imbalance is the sb but the raise
            # needs to be at least the bb
            min_raise = imbalance_size + max(imbalance_size, self.big_blind)
            if self.is_fixed_limit:
                raise_size = min_raise
            else:
                raise_size = player.choose_amount(minimum=min_raise,
                                                  maximum=player.stack,
                                                  pot_size=self.pot_size)
            player.bet_amount(raise_size)
            self.pot_size += raise_size
            amount_on_top = raise_size - imbalance_size
            # check if player is all in
            if player.stack == 0:
                if amount_on_top > 0:
                    self.update_hand_histories("{} raises {}, and is all-in\n"
                                               .format(player.name,
                                                       amount_on_top))
                else:
                    self.update_hand_histories("{} calls {}, and is all-in\n"
                                               .format(player.name,
                                                       raise_size))
                return False, True, amount_on_top
            self.update_hand_histories("{} raises {}\n".format(player.name,
                                                               amount_on_top))
            return False, False, amount_on_top
        elif choice == 'all-in':
            if other_player_is_all_in:
                all_in_amount = min(player.stack, imbalance_size)
            else:
                all_in_amount = player.stack
            player.bet_amount(all_in_amount)
            self.pot_size += all_in_amount
            amount_on_top = all_in_amount - imbalance_size
            if amount_on_top > 0:
                self.update_hand_histories("{} bets {}, and is all-in\n"
                                           .format(player.name, amount_on_top))
            else:
                self.update_hand_histories("{} calls {}, and is all-in\n"
                                           .format(player.name, all_in_amount))
            return False, True, amount_on_top

    def betting_round(self, stage='pre-flop'):
        """
        Flow control of each betting round during the hand.

        Keyword arguments:
            stage (str): stage the betting round is for (default 'pre-flop')
            possible entries are 'pre-flop', 'flop', 'turn', 'river'

        Returns:
            someone_has_folded (bool): flag to pass on whether someone folds
            someone_is_all_in (bool): flag to pass on whether someone is
            all in
        """
        if stage == 'pre-flop':
            imbalance_size = self.big_blind - self.small_blind
            action_cycle = cycle([self.playerSB, self.playerBB])
            is_action_on_bb_cycle = cycle([False, True])
        else:
            imbalance_size = 0
            action_cycle = cycle([self.playerBB, self.playerSB])
            is_action_on_bb_cycle = cycle([True, False])
        # initiate variables
        nb_actions = 0
        player = action_cycle.__next__()
        is_action_on_bb = is_action_on_bb_cycle.__next__()
        someone_has_gone_all_in = False
        while (nb_actions < 2) or (nb_actions >= 2 and imbalance_size > 0):
            # show information if user is human
            if isinstance(player, HumanPlayer):
                if is_action_on_bb:
                    logging.info("{}".format(self.hand_history_BB))
                else:
                    logging.info("{}".format(self.hand_history_SB))
            # set up possible actions based on context
            actions = self.get_possible_actions(player, imbalance_size,
                                                someone_has_gone_all_in,
                                                nb_actions)

            # get action from player, using context and hand history, execute
            # action and update attributes of betting round
            if is_action_on_bb:
                has_folded, is_all_in, imbalance_size = \
                    self.get_action_from_player(player, actions,
                                                imbalance_size,
                                                someone_has_gone_all_in,
                                                self.json_hand_hist_BB)
            else:
                has_folded, is_all_in, imbalance_size = \
                    self.get_action_from_player(player, actions,
                                                imbalance_size,
                                                someone_has_gone_all_in,
                                                self.json_hand_hist_SB)
            # if has folded, get out of betting round and attribute winnings
            if has_folded:
                player = action_cycle.__next__()
                player.win_pot(self.pot_size)
                self.update_hand_histories('{} wins the pot: +{}$'
                                           .format(player.name,
                                                   self.pot_size))
                return True, False
            # update variables
            someone_has_gone_all_in = someone_has_gone_all_in or is_all_in
            nb_actions += 1
            player = action_cycle.__next__()
            is_action_on_bb = is_action_on_bb_cycle.__next__()
        if someone_has_gone_all_in:
            # if someone has gone all-in, there may be an imbalance left and
            # the first player to have moved may have had more chips,
            # in that case he needs to get back from the pot
            if imbalance_size < 0:
                # player will always be the first mover
                # they can't be a negative imbalance if the person who
                # concludes the betting round has more chips
                player.get_back_from_pot(-imbalance_size)
                self.pot_size += imbalance_size
            return False, True
        return False, False

    def play(self):
        """
        Flow control of each hand being played. Includes launching betting
        rounds, controlling conditions for early ending and evaluating
        winner at showdown

        Returns:
            (None): if someone folds before showdown
        """
        # blinds
        logging.debug('{} is Big Blind'.format(self.playerBB.name))
        self.playerBB.bet_amount(self.big_blind)
        self.playerSB.bet_amount(self.small_blind)
        self.pot_size += self.big_blind + self.small_blind

        # players' private cards
        logging.debug('{} has {}'.format(self.playerBB.name,
                                         self.handBB.private_cards))
        logging.debug('{} has {}'.format(self.playerSB.name,
                                         self.handSB.private_cards))

        # first betting round, pre-flop
        someone_has_folded, someone_is_all_in = \
            self.betting_round(stage='pre-flop')
        if someone_has_folded:
            self.pot_size = 0  # reset pot size in case want to replay hand
            return None

        logging.debug('Flop comes {}'.format(self.flop))
        self.json_hand_hist_BB['community_cards'] += self.flop
        self.json_hand_hist_SB['community_cards'] += self.flop
        self.update_hand_histories("***** Dealing flop: {}\n"
                                   .format(self.flop))
        # integrate info
        self.handBB.add_public_cards(self.flop)
        self.handSB.add_public_cards(self.flop)
        # second betting round, post flop
        if not someone_is_all_in:
            someone_has_folded, someone_is_all_in = \
                self.betting_round(stage='flop')
            if someone_has_folded:
                self.pot_size = 0  # reset pot size in case want to replay hand
                return None

        logging.debug('Turn comes {}'.format(self.turn))
        self.json_hand_hist_BB['community_cards'] += self.turn
        self.json_hand_hist_SB['community_cards'] += self.turn
        self.update_hand_histories("***** Dealing turn: {} - {}\n"
                                   .format(self.flop, self.turn))
        # integrate info
        self.handBB.add_public_cards(self.turn)
        self.handSB.add_public_cards(self.turn)
        # Third betting round, post turn
        if not someone_is_all_in:
            someone_has_folded, someone_is_all_in = \
                self.betting_round(stage='turn')
            if someone_has_folded:
                self.pot_size = 0  # reset pot size in case want to replay hand
                return None

        logging.debug('River comes {}'.format(self.river))
        self.json_hand_hist_BB['community_cards'] += self.river
        self.json_hand_hist_SB['community_cards'] += self.river
        self.update_hand_histories("***** Dealing river: {} - {} - {}\n"
                                   .format(self.flop, self.turn, self.river))
        # integrate info
        self.handBB.add_public_cards(self.river)
        self.handSB.add_public_cards(self.river)
        # Fourth and last betting round, post river
        if not someone_is_all_in:
            someone_has_folded, someone_is_all_in = \
                self.betting_round(stage='river')
            if someone_has_folded:
                self.pot_size = 0  # reset pot size in case want to replay hand
                return None

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
                                           self.handBB.human_readable_rank(),
                                           self.playerSB.name,
                                           self.handSB.private_cards,
                                           self.handSB.best_combination,
                                           self.handSB.human_readable_rank(),
                                           self.pot_size))

        # evaluate winner at showdown
        winner = compare_two_hands(self.handBB, self.handSB)
        if winner == 'hand1':
            self.playerBB.win_pot(self.pot_size)
            self.update_hand_histories('{} wins the pot: +{}$'
                                       .format(self.playerBB.name,
                                               self.pot_size))
        elif winner == 'hand2':
            self.playerSB.win_pot(self.pot_size)
            self.update_hand_histories('{} wins the pot: +{}$'
                                       .format(self.playerSB.name,
                                               self.pot_size))
        else:
            self.playerBB.split_pot(self.pot_size)
            self.playerSB.split_pot(self.pot_size)
            self.update_hand_histories("Splitting the pot")

        self.pot_size = 0  # reset pot size in case want to replay hand
