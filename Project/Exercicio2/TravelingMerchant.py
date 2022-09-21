# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:00:22 2022

@author: Mihail Ababii
"""
import random
import numpy as np

from geneticalgorithm import GeneticAlgorithm as Genetic


class TravelingMerchant:
    def __init__(self, n_cities=4, population_size=1000):
        self.cities = []
        for _ in range(population_size):
            new_board = [(random.randint(0, 100), random.randint(0, 100), idx)
                         for idx in range(len(n_cities))]
            random.shuffle(new_board)
            self.cities.append(new_board)
        self.solution = self.cities[0].copy()
        self.distance = self.new_distance(self.solution)

    @staticmethod
    def new_distance(solution):
        distance = 0
        for i in range(1, len(solution)):
            x1, y1, idx1 = solution[i]
            x2, y2, idx2 = solution[i - 1]

            if idx1 == idx2:
                distance += np.inf
                break

            distx = (x1 - x2) ** 2
            disty = (y1 - y2) ** 2
            distance += np.sqrt(distx + disty)
        return distance

    @staticmethod
    def get_neighbours(solution):
        neighbours = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                neighbour = solution.copy()
                neighbour[i] = solution[j]
                neighbour[j] = solution[i]
                neighbours.append(neighbour)
        return neighbours

    def cost(self, neighbours):
        smallest_distance = self.new_distance(neighbours[0])
        selected_neighbour = neighbours[0]

        for neighbour in neighbours:
            distance = self.new_distance(neighbour)

            if distance < smallest_distance:
                smallest_distance = distance
                selected_neighbour = neighbour

        return selected_neighbour, smallest_distance

    def hill_climbing(self):
        current_route_length = self.distance
        neighbours = self.get_neighbours(self.solution)
        best_neighbour, best_neighbour_route_length = self.cost(neighbours)

        while best_neighbour_route_length < current_route_length:
            current_solution = best_neighbour
            current_route_length = best_neighbour_route_length
            neighbours = self.get_neighbours(current_solution)
            best_neighbour, best_neighbour_route_length = self.cost(neighbours)

        return current_solution, current_route_length

    def simulated_annealing(self, start_temperature=100000.0, alpha=0.99):
        current_solution = self.solution.copy()
        current_route_length = self.distance

        T = start_temperature
        while True:
            T = T * alpha
            if T <= 0.000001:
                return current_solution, current_route_length

            neighbours = self.get_neighbours(current_solution)
            idx = random.randint(1, len(neighbours))
            neighbour = neighbours[idx - 1]

            neighbour_route_length = self.new_distance(neighbour)
            if neighbour_route_length < current_route_length:
                current_solution = neighbour
                current_route_length = neighbour_route_length
            else:
                accept_p = np.exp((current_route_length - neighbour_route_length) / T)
                p = np.random.RandomState().random()
                if p < accept_p:
                    current_solution = neighbour
                    current_route_length = neighbour_route_length

    def genetic_algorithm(self):
        return Genetic(self.cities, self.new_distance).fit(fitness=-1)

    def show(self, current_solution, current_route_length):
        solution = []
        for i in range(len(current_solution)):
            x, y, idx = current_solution[i]
            solution.append(idx)
        print("Solution", solution)
        print("Route Length", current_route_length)

        print("-------------------------------------------")


def main():
    # n_cities = int(input("Enter Number of Cities: "))
    n_cities = 8
    homes = [[0] * n_cities for _ in range(n_cities)]

    for i in range(n_cities):
        for j in range(i, n_cities):
            if i != j:
                distance = random.uniform(0, 10)
                homes[i][j] = distance
                homes[j][i] = distance

    traveler = TravelingMerchant(homes, 8)
    current_solution, current_route_length = traveler.hill_climbing()
    traveler.show(current_solution, current_route_length)

    current_solution2, current_route_length2 = traveler.simulated_annealing()
    traveler.show(current_solution2, current_route_length2)
    current_solution3, current_route_length3 = traveler.genetic_algorithm()
    traveler.show(current_solution3, current_route_length3)


if __name__ == "__main__":
    main()
