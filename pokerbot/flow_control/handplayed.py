
import logging
from itertools import cycle

from ..hand_evaluation.hand import Hand, compare_two_hands
from ..utils import action_input, amount_input

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


class HandPlayed(object):
    """
    Hand being played object managing flow control for the hand played
    being two poker players, based on the current parameters of the game

    Attributes:
        playerBB (class.Player): player who will play big blind on this hand
        playerSB (class.Player): player who will play small blind on this hand
        pot_size (int): size of the pot on that hand
        big_blind (int): initial compulsory stake
        small_blind (int): second initial compulsory stake
        handBB (class.Hand): hand object of player on big blind
        handSB (class.Hand): hand object of player on small blind
        flop (list): list of communal cards coming on the flop
        turn (list): list containing communal card coming on the turn
        river (list): list containing communal card coming on the river
    """

    def __init__(self, player1, player2, big_blind, cards):
        """
        Instantiate a hand played object based on players, parameters,
        and list of 9 randomly drawn cards
        e.g. HandPlayed(Player(100,'Joe'), Player(100,'Mike'),
                        10, Deck().deal_cards(9))
        """
        self.playerBB = player1
        self.playerSB = player2
        self.pot_size = 0
        self.big_blind = big_blind
        self.small_blind = int(big_blind / 2)
        self.handBB = Hand([cards[0], cards[1]])
        self.handSB = Hand([cards[2], cards[3]])
        self.flop = [cards[4], cards[5], cards[6]]
        self.turn = [cards[7]]
        self.river = [cards[8]]

    def get_action_from_player(self, player, imbalance_size,
                               other_player_is_all_in):
        """
        Getting actions from players by prompting them for answers

        Args:
            player (class.Player): class object Player
            imbalance_size (int): pre action imbalance size, i.e.
            positive if one player has put more into the pot than the other
            other_player_is_all_in (bool): explicit, to set up maximum bet

        Returns:
            has_folded (bool): let know whether someone folded
            is_all_in (bool): let know whether someone is all in
            imbalance_size (int): post action imbalance size
        """
        logging.info('Action is on {}'.format(player.name))
        stack = player.stack
        logging.info('{} has a stack of {} $'.format(player.name, stack))

        if imbalance_size > 0:
            if imbalance_size >= stack:
                actions = ['all-in', 'fold']
            else:
                actions = ['call', 'raise', 'fold', 'all-in']
        else:
            actions = ['check', 'bet', 'fold', 'all-in']

        choice = action_input("Action?", actions)
        logging.info('{}\'s choice is: {}'.format(player.name, choice))

        if choice == 'fold':
            return True, False, imbalance_size
        elif choice == 'check':
            return False, False, 0
        elif choice == 'call':
            player.bet_amount(imbalance_size)
            self.pot_size += imbalance_size
            return False, False, 0
        elif choice == 'bet':
            if other_player_is_all_in:
                bet_size = amount_input("Amount?",
                                        minimum=self.big_blind,
                                        maximum=imbalance_size)
            else:
                bet_size = amount_input("Amount?", minimum=self.big_blind)
            player.bet_amount(bet_size)
            self.pot_size += bet_size
            return False, False, bet_size
        elif choice == 'raise':
            # minimum raise is calling the imbalance and doubling it
            # except pre-flop where the imbalance is the sb but the raise
            # needs to be at least the bb
            min_raise = imbalance_size + max(imbalance_size, self.big_blind)
            if other_player_is_all_in:
                raise_size = amount_input("Amount?",
                                          minimum=min_raise,
                                          maximum=imbalance_size)
            else:
                raise_size = amount_input("Amount?", minimum=min_raise)
            player.bet_amount(raise_size)
            self.pot_size += raise_size
            return False, False, raise_size - imbalance_size
        elif choice == 'all-in':
            if other_player_is_all_in:
                all_in_amount = min(stack, imbalance_size)
            else:
                all_in_amount = stack
            player.bet_amount(all_in_amount)
            self.pot_size += all_in_amount
            return False, True, all_in_amount - imbalance_size

    def betting_round(self, is_pre_flop=False):
        """
        Flow control of each betting round during the hand.

        Keyword arguments:
            is_pre_flop (bool): flag for post/pre flop (default False)

        Returns:
            someone_has_folded (bool): flag to pass on whether someone folds
            someone_is_all_in (bool): flag to pass on whether someone is
            all in
        """
        if is_pre_flop:
            imbalance_size = self.big_blind - self.small_blind
            action_cycle = cycle([self.playerSB, self.playerBB])
        else:
            imbalance_size = 0
            action_cycle = cycle([self.playerBB, self.playerSB])
        # initiate variables
        nb_actions = 0
        player = action_cycle.__next__()
        someone_has_gone_all_in = False
        while (nb_actions < 2) or (nb_actions >= 2 and imbalance_size > 0):
            # get action from player and update attributes of betting round
            has_folded, is_all_in, imbalance_size = \
                self.get_action_from_player(player,
                                            imbalance_size,
                                            someone_has_gone_all_in)
            # if has folded, get out of betting round and attribute winnings
            if has_folded:
                action_cycle.__next__().win_pot(self.pot_size)
                return True, False
            # update variables
            someone_has_gone_all_in = someone_has_gone_all_in or is_all_in
            nb_actions += 1
            player = action_cycle.__next__()
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
        logging.info('{} is Big Blind'.format(self.playerBB.name))
        self.playerBB.bet_amount(self.big_blind)
        self.playerSB.bet_amount(self.small_blind)
        self.pot_size += self.big_blind + self.small_blind

        # players' private cards
        logging.info('{} has {}'.format(self.playerBB.name,
                                        self.handBB.private_cards))
        logging.info('{} has {}'.format(self.playerSB.name,
                                        self.handSB.private_cards))

        # first betting round, pre-flop
        someone_has_folded, someone_is_all_in = \
            self.betting_round(is_pre_flop=True)
        if someone_has_folded:
            return None

        logging.info('Flop comes {}'.format(self.flop))
        # second betting round, post flop
        if not someone_is_all_in:
            someone_has_folded, someone_is_all_in = \
                self.betting_round(is_pre_flop=False)
            if someone_has_folded:
                return None

        logging.info('Turn comes {}'.format(self.turn))
        # Third betting round, post turn
        if not someone_is_all_in:
            someone_has_folded, someone_is_all_in = \
                self.betting_round(is_pre_flop=False)
            if someone_has_folded:
                return None

        logging.info('River comes {}'.format(self.river))
        # Fourth and last betting round, post river
        if not someone_is_all_in:
            someone_has_folded, someone_is_all_in = \
                self.betting_round(is_pre_flop=False)
            if someone_has_folded:
                return None

        # Evaluate winner at showdown
        # integrate info
        self.handBB.add_public_cards(self.flop + self.turn + self.river)
        self.handBB.update_best_combination()
        logging.info('{}\'s best combination is {}'
                     .format(self.playerBB.name, self.handBB.best_combination))
        self.handSB.add_public_cards(self.flop + self.turn + self.river)
        self.handSB.update_best_combination()
        logging.info('{}\'s best combination is {}'
                     .format(self.playerSB.name, self.handSB.best_combination))
        logging.info('{} has {}'.format(self.playerBB.name,
                                        self.handBB.human_readable_rank()))
        logging.info('{} has {}'.format(self.playerSB.name,
                                        self.handSB.human_readable_rank()))
        # evaluate winner at showdown
        winner = compare_two_hands(self.handBB, self.handSB)
        if winner == 'hand1':
            self.playerBB.win_pot(self.pot_size)
        elif winner == 'hand2':
            self.playerSB.win_pot(self.pot_size)
        else:
            self.playerBB.split_pot(self.pot_size)
            self.playerSB.split_pot(self.pot_size)
