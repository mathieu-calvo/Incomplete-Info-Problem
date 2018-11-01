
import csv
from itertools import islice

# come up with a list of all the straights possible with a 52-card deck
STRAIGHTS = [set(islice(range(2, 15), k, k + 5, 1))
             for k in range(len(range(2, 15)) - 4)]
# add the wheel - Ace to 5 straight
STRAIGHTS.append({2, 3, 4, 5, 14})

# face cards dictionary
FACE_CARDS_RANK_DICT = {14: 'A', 11: 'J', 12: 'Q', 13: 'K'}

# pretty character for suits
PRETTY_SUIT_DICT = {'C': '\u2667', 'D': '\u2662',
                    'H': '\u2661', 'S': '\u2664'}

# numerical value for suits
NUMERICAL_SUIT_DICT = {'C': 1, 'D': 2, 'H': 3, 'S': 4}

# human readable rankings
HUMAN_READABLE_RANKINGS = {1: "High card", 2: "One pair", 3: "Two pairs",
                           4: "Three of a kind", 5: "Straight", 6: "Flush",
                           7: "Full house", 8: "Four of a kind",
                           9: "Straight flush"}

# Probability that your private hand (two cards) will end up being the best
# hand - from http://www.natesholdem.com/pre-flop-odds.php#Qx
reader = csv.reader(open('pokerbot/pokerbot/datafiles/'
                         'preflop_prob_best_hand_showdown.csv', 'r'))
PRE_FLOP_WINNING_PROB = {}
for row in reader:
    k, v = row
    PRE_FLOP_WINNING_PROB[k] = float(v)

# sequence of actions Id
SEQUENCE_ACTIONS_ID = {
    "": 0,
    "F": 1,
    "B": 2,
    "C": 3,
    "BC": 4,
    "CC": 5,
    "BF": 6,
    "BB": 7,
    "CB": 8,
    "BBF": 9,
    "BBC": 10,
    "CBF": 11,
    "CBC": 12,
    "BBB": 13,
    "CBB": 14,
    "BBBF": 15,
    "BBBC": 16,
    "CBBF": 17,
    "CBBC": 18,
    "BBBB": 19,
    "CBBB": 20,
    "BBBBF": 21,
    "BBBBC": 22,
    "CBBBF": 23,
    "CBBBC": 24,
}
