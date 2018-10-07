
import logging

from .flow_control.headsupgame import HeadsUpGame
# from flow_control.deck import Deck
# from hand_evaluation.hand import Hand

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


def try_functionality():
    hu = HeadsUpGame(3, 10, 2, "Lulu", "Math")
    hu.start_game()
