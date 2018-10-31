
import logging

from .flow_control.headsupgame import HeadsUpGame
from .opponents.humanplayer import HumanPlayer
from .opponents.randomplayer import RandomPlayer
from .opponents.fishplayer import FishPlayer
from .opponents.fixedpolicyplayer import StartingHandPlayer, \
    StrengthHandPlayer

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


def run():
    starting_stack = 100
    big_blind = 10
    max_nb_hands = 10
    player_one = HumanPlayer(starting_stack, "Sapiens")
    player_two = RandomPlayer(starting_stack, "Hazard")
    player_three = FishPlayer(starting_stack, "Nemo")
    player_four = RandomPlayer(starting_stack, "Random")
    player_five = StartingHandPlayer(starting_stack, "Tight")
    player_six = StrengthHandPlayer(starting_stack, "Carlo")
    hu = HeadsUpGame(max_nb_hands, big_blind, player_six, player_five, True)
    hu.start_game()
    return hu
