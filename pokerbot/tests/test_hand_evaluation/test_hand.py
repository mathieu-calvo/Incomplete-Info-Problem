import pytest

from pokerbot import Card, Hand, evaluate_hand_ranking, \
    compare_two_hands, tie_breaking


hands = [
    (Card(5, "C"), Card(4, "C"), Card(3, "C"), Card(2, "C"), Card(14, "C")),
    (Card(3, "C"), Card(3, "S"), Card(3, "H"), Card(3, "D"), Card(5, "S")),
    (Card(5, "C"), Card(5, "H"), Card(4, "H"), Card(4, "C"), Card(5, "S")),
    (Card(14, "C"), Card(6, "C"), Card(3, "C"), Card(4, "C"), Card(5, "C")),
    (Card(6, "C"), Card(2, "C"), Card(3, "C"), Card(4, "C"), Card(5, "S")),
    (Card(2, "C"), Card(2, "H"), Card(2, "S"), Card(6, "C"), Card(5, "S")),
    (Card(14, "C"), Card(2, "C"), Card(14, "H"), Card(5, "C"), Card(5, "S")),
    (Card(14, "C"), Card(2, "C"), Card(3, "C"), Card(4, "C"), Card(4, "S")),
    (Card(14, "C"), Card(11, "C"), Card(3, "C"), Card(4, "C"), Card(5, "S"))
]

output = \
    [
        (9, [5]),
        (8, [3, 5]),
        (7, [5, 4]),
        (6, [14, 6, 5, 4, 3]),
        (5, [6]),
        (4, [2, 6, 5]),
        (3, [14, 5, 2]),
        (2, [4, 14, 3, 2]),
        (1, [14, 11, 5, 4, 3]),
    ]

test_eval_data = [[hands[i], output[i]] for i in range(9)]

test_compare_data = [
    [
        Hand([Card(5, "C"), Card(4, "C")]),
        Hand([Card(5, "S"), Card(4, "S")]),
        [Card(2, "H"), Card(3, "C"), Card(14,  "C")],
        'draw'
    ],
    [
        Hand([Card(5, "C"), Card(4, "C")]),
        Hand([Card(5, "S"), Card(4, "S")]),
        [Card(2, "C"), Card(3, "C"), Card(14, "C")],
        'hand1'
    ],
    [
        Hand([Card(14, "C"), Card(14, "H")]),
        Hand([Card(5, "S"), Card(4, "S")]),
        [Card(2, "C"), Card(3, "C"), Card(14, "C")],
        'hand2'
    ],
    [
        Hand([Card(14, "C"), Card(14, "H")]),
        Hand([Card(5, "S"), Card(4, "S")]),
        [Card(4, "C"), Card(4, "C"), Card(14, "C")],
        'hand1'
    ],
    [
        Hand([Card(14, "C"), Card(13, "H")]),
        Hand([Card(14, "S"), Card(12, "S")]),
        [Card(14, "D"), Card(4, "C"), Card(14, "H")],
        'hand1'
    ],
]

test_tiebreaking_data = [
    [
        [(Card(5, "C"), Card(4, "C"), Card(3, "C"),
          Card(2, "C"), Card(14, "C")),
         (Card(5, "C"), Card(4, "C"), Card(3, "C"),
          Card(2, "C"), Card(6, "C")),
         (Card(5, "C"), Card(4, "C"), Card(3, "C"),
          Card(6, "C"), Card(7, "C")),
         ],
        [[5], [6], [7]],
        [(Card(5, "C"), Card(4, "C"), Card(3, "C"),
          Card(6, "C"), Card(7, "C"))],
        [[7]]
    ],
    [
        [(Card(5, "C"), Card(5, "S"), Card(3, "C"),
          Card(3, "S"), Card(14, "C")),
         (Card(6, "C"), Card(6, "S"), Card(3, "C"),
          Card(3, "S"), Card(14, "C")),
         (Card(5, "C"), Card(5, "S"), Card(14, "C"),
          Card(3, "S"), Card(14, "C")),
         ],
        [[5, 3, 14], [6, 3, 14], [14, 5, 3]],
        [(Card(5, "C"), Card(5, "S"), Card(14, "C"),
          Card(3, "S"), Card(14, "C"))],
        [[14, 5, 3]]
    ]
]


@pytest.mark.parametrize("test_hand, expected_res", test_eval_data)
def test_evaluate_hand_ranking(test_hand, expected_res):
    assert evaluate_hand_ranking(test_hand) == expected_res


@pytest.mark.parametrize("test_hand1, test_hand2, public_cards, expected_res",
                         test_compare_data)
def test_compare_two_hands(test_hand1, test_hand2, public_cards, expected_res):
    test_hand1.add_public_cards(public_cards)
    test_hand1.update_best_combination()
    test_hand2.add_public_cards(public_cards)
    test_hand2.update_best_combination()
    assert compare_two_hands(test_hand1, test_hand2) == expected_res


@pytest.mark.parametrize("test_hands, test_tiebreakers, best_hand, "
                         "best_tiebreaker", test_tiebreaking_data)
def test_tie_breaking(test_hands, test_tiebreakers,
                      best_hand, best_tiebreaker):
    assert tie_breaking(test_hands, test_tiebreakers) == (best_hand,
                                                          best_tiebreaker)

