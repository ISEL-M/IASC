from random import random

import numpy as np
import random


class Selection:
    def __init__(self, populations, fitness_fn):
        self.populations = populations
        self.fitness_fn = fitness_fn

    def random(self):
        idx = random.randint(1, len(self.populations))
        x = self.populations[idx - 1]
        return x


class Mutation:
    def __init__(self, element):
        self.element = element

    def bit_flip(self):
        idx = random.randint(0, len(self.element) - 1)
        new_value = random.randint(1, len(self.element))
        self.element[idx] = new_value
        return self.element

    def swap(self):
        idx1 = random.randint(1, len(self.element)) - 1
        idx2 = random.randint(1, len(self.element)) - 1

        temp = self.element[idx1]
        self.element[idx1] = self.element[idx2]
        self.element[idx2] = temp

        return self.element


class Crossover:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def one_point(self):
        idx = random.randint(0, len(self.x) - 1)
        child = self.x[:idx] + self.y[idx:]
        return child


class GeneticAlgorithm:
    def __init__(self, population, fitness_function):
        self.population = population.copy()
        self.fitness_fn = fitness_function

    def fit(self, fitness, repeat=100000, mutation_p=0.03):
        population = self.population.copy()

        best_fit = self.population[0].copy()
        best_fitness = self.fitness_fn(best_fit)

        t = 1
        while t < repeat and best_fitness != fitness:
            new_population = []
            for _ in range(len(population)):
                x = Selection(population, self.fitness_fn).random()
                y = Selection(population, self.fitness_fn).random()

                child = Crossover(x, y).one_point()

                p = np.random.RandomState().random()
                if p < mutation_p:
                    child = Mutation(child).swap()

                new_population.append(child)

            for x in population:
                fitness = self.fitness_fn(x)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_fit = x

            t += 1

            population = new_population

        return best_fit, best_fitness
