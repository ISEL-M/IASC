import abc
import numpy as np

from multipledispatch import dispatch
from numpy import argmax
from numpy.random import shuffle, choice, random


class World_Map:
    def __init__(self, x_dim, y_dim, positions):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.positions = positions

    @staticmethod
    def get_start(world_map):
        for y in range(world_map.y_dim):
            for x in range(world_map.x_dim):
                if world_map.positions[(x, y)] == ">":
                    return State(x, y)

    @staticmethod
    def get_end(world_map):
        for y in range(world_map.y_dim):
            for x in range(world_map.x_dim):
                if world_map.positions[(x, y)] == "A":
                    return State(x, y)

    @staticmethod
    def convert_to_map(world_map):
        positions = {}
        x_dim = len(world_map)
        y_dim = len(world_map[1])
        for y in range(y_dim):
            for x in range(x_dim):
                positions[(x, y)] = world_map[x][y]

        return World_Map(x_dim, y_dim, positions)


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equals(self, state):
        return state.x == self.x and state.y == self.y

    def get(self):
        return self.x, self.y


class Action:
    def __init__(self, operation):
        self.operation = operation

    def apply(self, state: State):
        return self.operation(state)

    @staticmethod
    def cost(state: State, next_state: State):
        return np.sqrt(((state.x - next_state.x) ** 2 + (state.y - next_state.y) ** 2))

    @staticmethod
    def dir4():
        operators = {"up": Action(lambda state: State(state.x, state.y - 1)),
                     "down": Action(lambda state: State(state.x, state.y + 1)),
                     "right": Action(lambda state: State(state.x + 1, state.y)),
                     "left": Action(lambda state: State(state.x - 1, state.y))}

        return operators

    @staticmethod
    def dir8():
        operators = Action.dir4()
        operators["up_right"] = Action(lambda state: State(state.x + 1, state.y - 1))
        operators["down_right"] = Action(lambda state: State(state.x + 1, state.y + 1))
        operators["up_left"] = Action(lambda state: State(state.x - 1, state.y - 1))
        operators["down_left"] = Action(lambda state: State(state.x - 1, state.y + 1))

        return operators


class Learned_Memory:
    @abc.abstractmethod
    def Q(self, s: State, a: Action):
        pass

    @abc.abstractmethod
    def update(self, s: State, a: Action, p: float):
        pass


class Sparse_Memory(Learned_Memory):
    def __init__(self, default_value: float = 0.0):
        self.default_value = default_value
        self.memory = {}

    def Q(self, s: State, a: Action):
        return self.memory.get((s, a), self.default_value)

    def update(self, s: State, a: Action, p: float):
        self.memory[(s, a)] = p


class Sel_Action(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select_action(self, s: State):
        pass


class EGreedy(Sel_Action):
    def __init__(self, learned_mem: Learned_Memory, actions: Action, epsilon: float):
        self.learned_mem = learned_mem
        self.actions = actions
        self.epsilon = epsilon

    def max_action(self, s: State):
        shuffle(self.actions)
        return argmax(self.actions, lambda a: self.learned_mem.Q(s, a))

    def reuse_action(self, s: State):
        return self.max_action(s)

    def explore(self):
        return choice(self.actions)

    def select_action(self, s: State):
        if random() > self.epsilon:
            return self.reuse_action(s)
        else:
            return self.explore()


class Reinforcement_Learning:
    @dispatch(Sparse_Memory, EGreedy, float, float)
    def __init__(self, learned_mem: Learned_Memory, sel_action: EGreedy, alfa: float, gama: float):
        self.learned_mem = learned_mem
        self.sel_action = sel_action
        self.alfa = alfa
        self.gama = gama

    @abc.abstractmethod
    def learn(self, s: State, a: Action, r: float, sn: State, an: Action = None):
        pass


class QLearning(Reinforcement_Learning):
    def learn(self, s: State, a: Action, r: float, sn: State, an: Action = None):
        an = self.sel_action.max_action(sn)
        qsa = self.learned_mem.Q(s, a)
        qsnan = self.learned_mem.Q(sn, an)
        q = qsa + self.alfa * (r + (self.gama * qsnan) - qsa)
        self.learned_mem.update(s, a, q)


class DynaQ(QLearning):
    @dispatch(Sparse_Memory, EGreedy, float, float, int)
    def __init__(self, learned_mem: Learned_Memory, sel_action: Sel_Action, alfa: float, gama: float, sim_num: int):
        self.__init__(learned_mem, sel_action, alfa, gama)
        self.sim_num = sim_num
        self.model = World()

    def learn(self, s: State, a: Action, r: float, sn: State):
        self.learn(s, a, r, sn)
        self.model.update(s, a, r, sn)
        self.simulate()

    def simulate(self):
        for _ in range(self.sim_num):
            s, a, r, sn = self.model.show()
            self.learn(s, a, r, sn)


class World:
    def __init__(self):
        self.T = {}
        self.R = {}

    def update(self, s: State, a: Action, r: float, sn: State):
        self.T[(s, a)] = sn
        self.R[(s, a)] = r

    def show(self):
        keys = list(self.T.keys())
        s, a = choice(keys)
        sn = self.T[(s, a)]
        r = self.R[(s, a)]

        return s, a, r, sn


if __name__ == "__main__":
    memory = Sparse_Memory()
    sel_action = EGreedy(memory, Action.dir4(), 0.5)
    dyna_q = DynaQ(memory, sel_action, 0.1, 0.95, 10000)

    dyna_q.learn()
