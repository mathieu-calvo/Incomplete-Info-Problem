
from .dqnagent import DQNAgent
from ..flow_control.headsupgame import HeadsUpGame
from ..flow_control.handplayed import HandPlayed


class PokerEnv(object):
    """
    Instantiating the object returns an environment where a heads-up game is
    being played between an agent and a given player

    Attributes:
        agent (DQNAgent): DQN agent, player one
        opponent (Player.subclass):  player two, the opponent
    """

    def __init__(self, opponent):
        self.agent = DQNAgent()
        self.opponent = opponent
        self.game = HeadsUpGame(10, 10, self.agent, self.opponent, True)

    def step(self, action):
        next_state, reward, done, info = HandPlayed().step(action)
        return next_state, reward, done, info

    def reset(self):
        self.game = HeadsUpGame(10, 10, self.agent, self.opponent, True)
