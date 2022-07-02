import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font("./arial.ttf", 25)
# font = pygame.font.SysFont("arial", 25)

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")
BLOCK_SIZE = 20
SPEED = 80
WHITE = (255, 255, 255)
GRAY = (121, 121, 121)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class SnakeGameAI:
    def __init__(self, w=640, h=480) -> None:
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.past_head_locations = []
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):

        self.display.fill(BLACK)
        pygame.draw.rect(
            self.display,
            WHITE,
            pygame.Rect(self.snake[0].x, self.snake[0].y, BLOCK_SIZE, BLOCK_SIZE),
        )
        pygame.draw.rect(
            self.display,
            GRAY,
            pygame.Rect(
                self.snake[0].x, self.snake[0].y, BLOCK_SIZE * 0.75, BLOCK_SIZE * 0.75
            ),
        )
        for pt in self.snake[1:]:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display,
                BLUE2,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE * 0.75, BLOCK_SIZE * 0.75),
            )
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        text = font.render("Score : {}".format(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _move(self, action):
        # 1 0 0 - straight
        # 0 1 0 - right
        # 0 0 1 - left
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [0, 0, 1]):
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        elif np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]

        if len(self.snake) < 4:
            self.past_head_locations.append(self.head)
            # only care about rotations till its 3 block long

        self.direction = new_dir
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False

        # if len(self.past_directions) > 4:
        #     if self.past_directions[-5:] == 5 * [
        #         Direction.RIGHT
        #     ] or self.past_directions[-5:] == 5 * [Direction.LEFT]:
        #         reward = -5
        #         print("Rotating with: ", self.past_directions)
        #     else:
        #         self.past_directions = self.past_directions[-5:]
        if len(self.snake) < 4:
            if len(self.past_head_locations) >= 4:
                if self.past_head_locations[-1] == self.past_head_locations[-4]:
                    reward += -10
                    print("You're looping mate")
                else:
                    self.past_head_locations = self.past_head_locations[-4:]

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward += -20
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward += 100
            self._place_food()
        else:
            self.snake.pop()
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
