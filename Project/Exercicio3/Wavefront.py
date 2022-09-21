import time
from os import listdir
from multipledispatch import dispatch
import numpy as np

from Display import *


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


class Operator:
    def __init__(self, operation):
        self.operation = operation

    def apply(self, state: State):
        return self.operation(state)

    def cost(self, state: State, next_state: State):
        return np.sqrt(((state.x - next_state.x) ** 2 + (state.y - next_state.y) ** 2))

    @staticmethod
    def dir4():
        operator1 = Operator(lambda state: State(state.x, state.y + 1))
        operator2 = Operator(lambda state: State(state.x, state.y - 1))
        operator3 = Operator(lambda state: State(state.x + 1, state.y))
        operator4 = Operator(lambda state: State(state.x - 1, state.y))

        return operator1, operator2, operator3, operator4

    @staticmethod
    def dir8():
        operator1, operator2, operator3, operator4 = Operator.dir4()
        operator5 = Operator(lambda state: State(state.x + 1, state.y + 1))
        operator6 = Operator(lambda state: State(state.x + 1, state.y - 1))
        operator7 = Operator(lambda state: State(state.x - 1, state.y + 1))
        operator8 = Operator(lambda state: State(state.x - 1, state.y - 1))

        return operator1, operator2, operator3, operator4, operator5, operator6, operator7, operator8


class Node:
    @dispatch(State)
    def __init__(self, state: State):
        self.state = state
        self.depth = 0
        self.cost = 0

    @dispatch(State, Operator, object)
    def __init__(self, state: State, operator: Operator, previous_node: 'Node'):
        self.state = state
        self.previous_node = previous_node
        self.operator = operator
        self.depth = previous_node.depth + 1
        self.cost = operator.cost(previous_node.state, state)


class Problem:
    def __init__(self, initial_state: State, end_state: State, operators: Operator):
        self.initial_state = initial_state
        self.end_state = end_state
        self.operators = operators

    def objective(self, state: State):
        return self.end_state.equals(state)

    def heuristic(self):
        pass


class Memory:
    def __init__(self, world: World_Map):
        self.world = world
        self.search = []
        self.known = []

    def clean(self):
        self.search = []
        self.known = []

    def insert(self, new_node: Node):
        for node in self.known:
            if node.state.equals(new_node.state):
                return
        for node in self.search:
            if node.state.equals(new_node.state):
                return
        self.search.append(new_node)

    def remove(self):
        node = self.search.pop(0)
        self.known.append(node)
        return node

    def is_complete(self):
        return not self.search


class Wavefront:
    def __init__(self, world: World_Map, display=None):
        self.display = display
        self.world = world
        self.memory_search = Memory(world)

    @dispatch(Problem, float)
    def solve(self, problem: Problem, prof_max):
        self.memory_search.clean()
        no_initial = Node(problem.initial_state)
        self.memory_search.insert(no_initial)
        while not self.memory_search.is_complete():
            node = self.memory_search.remove()
            if problem.objective(node.state):
                return self.get_solution(node)
            else:
                if node.depth < prof_max:
                    self.expand(node, problem)

    @dispatch(Problem)
    def solve(self, problem: Problem):
        return self.solve(problem, np.inf)

    def expand(self, node: Node, problem: Problem):
        state = node.state
        for operator in problem.operators:
            state_suc = operator.apply(state)
            if self.world.positions[state_suc.get()] != "O":
                next_node = Node(state_suc, operator, node)
                self.memory_search.insert(next_node)

    def get_solution(self, node: Node):
        solution = {}
        cost = 0
        depth = 0
        if hasattr(node, "previous_node"):
            solution, cost, depth = self.get_solution(node.previous_node)
            solution[node.previous_node.state.get()] = (node.operator, node.depth, node.cost)
            if depth < node.depth:
                depth = node.depth
            cost += node.cost
        return solution, cost, depth

    def get_solution(self, node: Node):
        solution = {}
        cost = 0
        depth = 0
        if hasattr(node, "previous_node"):
            solution, cost, depth = self.get_solution(node.previous_node)
            solution[node.previous_node.state.get()] = (node.operator, node.depth, node.cost)
            if depth < node.depth:
                depth=node.depth
            cost += node.cost
        return solution, cost, depth

    def display(self):
        self

if __name__ == "__main__":
    def load_map(file):
        f = open("maps/" + file, "r")
        map = []
        for _, l in enumerate(f):
            line = []
            for i in range(len(l) - 1):
                line.append(l[i])
            map.append(line)
        f.close()
        return map

    def get_maps():
        files = listdir('maps/')
        print("Available Maps:")
        idx = 0
        for f in files:
            print("\t", idx, "-", f.split(".")[0])
            idx += 1

        return files

    def select_map():
        maps_names = get_maps()
        val = input("Select Map: ")
        idx = int(val)
        return maps_names[idx]

    def set_display(world_map):
        display = Display(world_map.x_dim, world_map.y_dim)
        for position in world_map.positions:
            if world_map.positions[position] == "O":
                display.set_node(position[0], position[1], BLACK)
            if world_map.positions[position] == ">":
                display.set_node(position[0], position[1], RED)
            if world_map.positions[position] == "A":
                display.set_node(position[0], position[1], GREEN)

        return display

    def show(display, solution, depth):
        state = initial_state
        for i in range(depth):
            operation, _, _ = solution[state.get()]
            state = operation.apply(state)
            if i == depth-1:
                display.set_node(state.x, state.y, PURPLE)
            else:
                display.set_node(state.x, state.y, YELLOW)
            display.draw()
            time.sleep(0.2)
        time.sleep(10)

    selected_map = select_map()
    world_map = load_map(selected_map)
    world_map = World_Map.convert_to_map(world_map)
    # mapOfLand.display()

    print()
    initial_state = World_Map.get_start(world_map)
    print("Initial State", initial_state.get())
    end_state = World_Map.get_end(world_map)
    print("Goal State", end_state.get())

    problem4 = Problem(initial_state, end_state, Operator.dir4())
    problem8 = Problem(initial_state, end_state, Operator.dir8())

    solution4, cost4, depth4 = Wavefront(world=world_map).solve(problem4)
    display4 = set_display(world_map)
    show(display4, solution4, depth4)

    print("cost->", cost4)
    print("depth->", depth4)
    print("-----------------")
    solution8, cost8, depth8 = Wavefront(world=world_map).solve(problem8)
    display8 = set_display(world_map)
    show(display8, solution8, depth8)
    print("cost->", cost8)
    print("depth->", depth8)

