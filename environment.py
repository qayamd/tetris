import numpy as np
import random
from collections import deque

class Tetris:
    def __init__(self, height=20, width=10):
        self.height = height
        self.width = width
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.game_over = False
        self.shapes = [
            [[1, 1, 1, 1]],  # I
            [[1, 1], [1, 1]],  # O
            [[0, 1, 0], [1, 1, 1]],  # T
            [[1, 0], [1, 0], [1, 1]],  # J
            [[0, 1], [0, 1], [1, 1]]  # L
            [[1, 1, 0], [0, 1, 1]],  # S
            [[0, 1, 1], [1, 1, 0]]  # Z
        ]
        self.current_piece = self.get_new_piece()
        self.current_position = {'x': self.width // 2 - len(self.current_piece[0]) // 2, 'y': 0}
        
    def get_new_piece(self):
        return random.choice(self.shapes)
    
    def rotate_piece(self, piece):
        return [list(reversed(col)) for col in zip(*piece)]
    
    def step(self, action):
        rotation, x = action
        piece = self.current_piece
        for _ in range(rotation):
            piece = self.rotate_piece(piece)
        
        if self.valid_position(piece, x, 0):
            self.current_position = {'x': x, 'y': 0}
            while self.valid_position(piece, self.current_position['x'], self.current_position['y'] + 1):
                self.current_position['y'] += 1
            self.place_piece(piece)
            self.clear_lines()
            if not self.game_over:
                self.current_piece = self.get_new_piece()
                self.current_position = {'x': self.width // 2 - len(self.current_piece[0]) // 2, 'y': 0}
                if not self.valid_position(self.current_piece, self.current_position['x'], self.current_position['y']):
                    self.game_over = True
            return self.calculate_reward()
        else:
            return -10  # Invalid move punishment

    def valid_position(self, piece, x, y):
        for i in range(len(piece)):
            for j in range(len(piece[0])):
                if piece[i][j]:
                    if i + y >= self.height or j + x < 0 or j + x >= self.width or self.board[i + y][j + x]:
                        return False
        return True

    def place_piece(self, piece):
        for i in range(len(piece)):
            for j in range(len(piece[0])):
                if piece[i][j]:
                    self.board[self.current_position['y'] + i][self.current_position['x'] + j] = 1

    def clear_lines(self):
        lines_to_clear = [i for i in range(self.height) if all(self.board[i])]
        for line in lines_to_clear:
            self.board = np.delete(self.board, line, 0)
            self.board = np.insert(self.board, 0, np.zeros((1, self.width)), axis=0)
    
    def calculate_reward(self):
        lines_cleared = sum(1 for i in range(self.height) if all(self.board[i]))
        holes = sum(1 for i in range(self.height) for j in range(self.width) if self.board[i][j] == 0 and i > 0 and self.board[i-1][j] == 1)
        height = max((self.height - i for i in range(self.height) if any(self.board[i])), default=0)
        return lines_cleared * 10 - holes * 2 - height * 0.5

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.game_over = False
        self.current_piece = self.get_new_piece()
        self.current_position = {'x': self.width // 2 - len(self.current_piece[0]) // 2, 'y': 0}
        return self.get_state()

    def get_state(self):
        board = self.board.flatten()
        holes = sum(1 for i in range(self.height) for j in range(self.width) if self.board[i][j] == 0 and i > 0 and self.board[i-1][j] == 1)
        height = max((self.height - i for i in range(self.height) if any(self.board[i])), default=0)
        return np.concatenate([board, [holes, height]])
