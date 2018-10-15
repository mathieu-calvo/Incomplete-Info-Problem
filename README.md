# Incomplete Info Problem
This research project will aim at tackling an incomplete information game using reinforcement
learning techniques.

In this project, we initially aim at targeting arguably the most popular incomplete information
card game at the moment, the very famous game of poker. Specifically, we will only consider its
most common variant, also known as the Limit Texas Hold’em Poker game

“The challenges introduced by poker are many. The game involves a number of forms of
*uncertainty*, including stochastic dynamics from a shuffled deck, *imperfect 
information* due to
the opponent’s private cards, and, finally, an *unknown opponent*. These 
uncertainties are
individually difficult and together the difficulties only escalate. A related challenge is the
problem of *folded hands*, which amount to partial observations of the 
opponent’s decision
making contexts. (…). A third key challenge is the *high variance of 
payoffs*, also known as luck.
This makes it difficult for a program to even assess its performance over short periods of time.
To aggravate this difficulty, play against human opponents is necessarily limited. If no more than
two or three hundred hands are to be played in total, opponent modelling must be effective using
only very small amounts of data. Finally, Texas hold’em is a very large game. It has on the order
of 10^18 states, which makes even straightforward calculations, such as best
 response, nontrivial.”
[Southey, F., Bowling, M. P., Larson, B., Piccione, C., Burch, N., Billings, D., & Rayner,
C. (2012). Bayes' bluff: Opponent modelling in poker. arXiv preprint 
arXiv:1207.1411]

# Project scope
In order to limit the complexity of this research project, we will proceed to a few simplifications.
* We will limit our analysis to one versus one poker games, also known as 
“heads-up” poker games.
* Our poker-playing agent will specialize in playing games where the initial
 stakes do not
change over the course of the session, also known as “cash-games”.
* We will assume that only a limited number of bet sizes are possible at 
each point in time
* We will assume that each session ends after a given number of hands.
* And most importantly, the initial opponent of our poker-playing agent will
 be limited to a simple reflex-agent that follows a fixed policy
 
# The challenges
This research project will entail several challenges:
* Write a poker game framework programmatically, including flow control, 
card shuffling
and the sequence of events.
* Retrieve the poker game statistical knowledge, with hand values plus hand 
potentials
(forward-looking) at each betting round based on available information.
* Design a model of opponent’s possible range of hands given its action 
during the hand and during the session. This will involve a matrix of all possible combinations (excluding
impossible ones, inferred from visible cards). This should allow for a probability
estimation of our opponent bluffing.
* Design a decision process model that take many factors into account: 
position, players' stacks (number of chips), hand value and hand potential, opponent modelling
* Maximize quality of decisions knowing that good decisions could often lead
 to losses, and especially because our agent will not necessarily know what was the quality of its
decisions (because only the showdown reveals the full state of the environment)