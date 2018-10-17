
import logging

from .flow_control.headsupgame import HeadsUpGame
from .opponents.humanplayer import HumanPlayer
from .opponents.randomplayer import RandomPlayer
from .opponents.fishplayer import FishPlayer

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


def run():
    starting_stack = 500
    big_blind = 20
    max_nb_hands = 10
    player_one = HumanPlayer(starting_stack, "Sapiens")
    player_two = RandomPlayer(starting_stack, "Hazard")
    player_three = FishPlayer(starting_stack, "Nemo")
    player_four = RandomPlayer(starting_stack, "Random")
    hu = HeadsUpGame(max_nb_hands, big_blind, player_four, player_two)
    hu.start_game()
    return hu
