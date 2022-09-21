# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:00:22 2022

@author: Mihail Ababii
"""
import random
import timeit
import numpy as np

from geneticalgorithm import GeneticAlgorithm as Genetic


class NQueen:
    def __init__(self, n_queens=8, population_size=1000):
        self.boards = []
        for _ in range(population_size):
            new_board = [y + 1 for y in range(n_queens)]
            random.shuffle(new_board)
            self.boards.append(new_board)

        self.board = self.boards[0].copy()
        self.curr_cost = self.cost(self.board)

    @staticmethod
    def show(solution, cost):
        print("solution:", solution)

        nq = len(solution)
        board = [["x"] * nq for _ in range (nq)]
        for i in range(nq):
            board[solution[i]-1][i] = "Q"
        for row in board:
            print("\t\t\t\t"," ".join(row))

        print("cost:", cost)

    @staticmethod
    def get_neighbours(board):
        neighbours = []
        for i in range(len(board)):
            for j in range(i + 1, len(board)):
                neighbour = board.copy()
                neighbour[i] = board[j]
                neighbour[j] = board[i]
                neighbours.append(neighbour)
        return neighbours

    def cost(self, board):
        cost = 0
        for i in range(len(board)):
            for j in range(len(board)):
                if i!=j:
                    difx = abs(i - j)
                    dify = abs(board[i] - board[j])
                    if difx == dify:
                        cost += 1
                    if board[i] == board[j]:
                        cost += 1

        return cost

    def best_cost_neighbour(self, neighbours):
        best_cost = self.cost(neighbours[0])
        best_neighbour = neighbours[0]
        for neighbour in neighbours:
            cost = self.cost(neighbour)
            if cost < best_cost:
                best_cost = cost
                best_neighbour = neighbour
        return best_neighbour, best_cost

    def hill_climbing(self):
        current_cost = self.curr_cost
        current_solution = self.board.copy()

        neighbours = self.get_neighbours(self.board)
        best_neighbour, best_cost = self.best_cost_neighbour(neighbours)

        while best_cost < current_cost:
            current_cost = best_cost
            current_solution = best_neighbour

            neighbours = self.get_neighbours(current_solution)
            best_neighbour, best_cost = self.best_cost_neighbour(neighbours)

        return current_solution, current_cost

    def simulated_annealing(self, start_temperature=100000.0, alpha=0.999):
        current_solution = self.board.copy()
        current_cost = self.curr_cost

        T = start_temperature
        while True:
            T = T * alpha
            if T <= 0.000001:
                return current_solution, current_cost

            neighbours = self.get_neighbours(current_solution)
            idx = random.randint(1, len(neighbours))
            neighbour = neighbours[idx - 1]

            best_cost = self.cost(neighbour)
            if best_cost < current_cost:
                current_solution = neighbour
                current_cost = best_cost
            else:
                accept_p = np.exp((current_cost - best_cost) / T)
                p = np.random.RandomState().random()
                if p < accept_p:
                    current_solution = neighbour
                    current_cost = best_cost

    def genetic_algorithm(self):
        return Genetic(self.boards, self.cost).fit(fitness=0)


def main():
    nq = int(input("Enter Number of Queens: "))
    n_queens = NQueen(n_queens=nq, population_size=1000)

    start = timeit.default_timer()
    print("Hill-Climbing")
    solution, cost = n_queens.hill_climbing()
    n_queens.show(solution, cost)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    start = timeit.default_timer()
    print()
    print("Simulated-annealing")
    solution, cost = n_queens.simulated_annealing()
    n_queens.show(solution, cost)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    start = timeit.default_timer()
    print()
    print("Genetic Algorithm")
    solution, cost = n_queens.genetic_algorithm()
    n_queens.show(solution, cost)
    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == "__main__":
    main()
