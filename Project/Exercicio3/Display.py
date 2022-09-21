import threading
import time

import pygame

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Node:
    def __init__(self, row, col, gap_x, gap_y):
        self.row = row
        self.col = col

        self.x = row * gap_x
        self.y = col * gap_y

        self.gap_x = gap_x
        self.gap_y = gap_y

        self.color = WHITE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.gap_x, self.gap_y))

    def set_color(self, color):
        self.color = color


class Display:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.max_x = int(900 / self.x) * self.x
        self.max_y = int(700 / self.y) * self.y

        self.gap_x = self.max_x // self.x
        self.gap_y = self.max_y // self.y

        self.grid = []

        self.win = pygame.display.set_mode((self.max_x, self.max_y))
        self.win.fill((255, 255, 255))
        pygame.display.set_caption("Exercise 3")
        self.make_grid()
        self.draw_grid()

    def make_grid(self):
        for i in range(self.x):
            self.grid.append([])
            for j in range(self.y):
                node = Node(i, j, self.gap_x, self.gap_y)
                self.grid[i].append(node)

    def draw_grid(self):
        for i in range(1, self.x + 1):
            pygame.draw.line(self.win, GREY, (0, i * self.gap_y), (self.max_x, i * self.gap_y))
        for j in range(1, self.y + 1):
            pygame.draw.line(self.win, GREY, (j * self.gap_x, 0), (j * self.gap_x, self.max_y))
        pygame.display.update()

    def draw(self):
        self.win.fill(WHITE)

        for row in self.grid:
            for spot in row:
                spot.draw(self.win)
        self.draw_grid()

        pygame.display.update()

    def set_node(self, x, y, color):
        node = self.grid[y][x]
        node.set_color(color)
        pygame.draw.rect(self.win, node.color, (x * self.gap_x, y * self.gap_y, self.gap_x, self.gap_y))
        pygame.display.update()



if __name__ == "__main__":
    display = Display(7, 6)
    display.set_node(0, 0, RED)
    time.sleep(10)

