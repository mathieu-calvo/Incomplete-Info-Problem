
import random

from ..flow_control.deck import Deck
from ..hand_evaluation.hand import Hand, compare_two_hands


# Estimate the ratio of winning games given the current state of the game
def estimate_win_rate(nb_simulations, hole_cards, community_cards=None):
    """
    Estimate the win rate of a given hand, given the community cards,
    estimation is done with Monte Carlo simulations

    Args:
        nb_simulations (int): number of MC simulations
        hole_cards (list): list of two Card objects
        community_cards (list): list of Card objects, representing board
        cards, default is an empty list

    Returns:
        (float): win rate estimated using MC simulations
    """
    # default community cards to empty list
    if community_cards is None:
        community_cards = []
    # estimate the win count by doing Monte Carlo simulation,
    win_count = sum([monte_carlo_simulation(hole_cards, community_cards)
                     for _ in range(nb_simulations)])
    return 1.0 * win_count / nb_simulations


def monte_carlo_simulation(hole_cards, community_cards):
    """
    Estimate the win rate of a given hand, given randomly drawn missing
    community cards, and randomly drawn opponent hole_cards. Estimation is
    done with Monte Carlo simulations

    Args:
        hole_cards (list): list of two Card objects
        community_cards (list): list of Card objects, representing board
        cards, default is an empty list

    Returns:
        (bool): 1 if hero hand wins or draws, 0 otherwise, given estimated
        MC randomly drawn parameters
    """
    # start from remaining cards
    remaining_cards = Deck().get_remaining_cards(hole_cards + community_cards)

    # draw missing community cards randomly
    nb_missing_community_cards = 5 - len(community_cards)
    missing_community_cards = random.sample(remaining_cards,
                                            nb_missing_community_cards)
    remaining_cards = [card for card in remaining_cards
                       if card not in missing_community_cards]

    # draw opponent cards randomly
    opponent_hole_cards = random.sample(remaining_cards, 2)

    # update hands accordingly
    public_cards = community_cards + missing_community_cards
    opponent_hand = Hand(opponent_hole_cards)
    opponent_hand.add_public_cards(public_cards)
    opponent_hand.update_best_combination()
    hero_hand = Hand(hole_cards)
    hero_hand.add_public_cards(public_cards)
    hero_hand.update_best_combination()

    # evaluate winner
    winner = compare_two_hands(hero_hand, opponent_hand)
    if winner == 'hand2':
        return 0
    else:
        return 1
