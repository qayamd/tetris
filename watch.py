import torch
import numpy as np
import pygame
import random
from dqn import DeepQNetwork

class Tetris:
    pieceColors = [
        (0, 0, 0), (255, 255, 0), (147, 88, 254), (54, 175, 144), 
        (255, 0, 0), (102, 217, 238), (254, 151, 32), (0, 0, 255)
    ]

    pieces = [
        [[1, 1], [1, 1]], [[0, 2, 0], [2, 2, 2]], 
        [[0, 3, 3], [3, 3, 0]], [[4, 4, 0], [0, 4, 4]], 
        [[5, 5, 5, 5]], [[0, 0, 6], [6, 6, 6]], 
        [[7, 0, 0], [7, 7, 7]]
    ]

    def __init__(self, height=20, width=10, block_size=20):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extraBoard = np.ones((height * block_size, width * block_size // 2, 3), dtype=np.uint8) * 204
        self.text_color = (200, 20, 220)
        pygame.init()
        self.screen = pygame.display.set_mode((width * block_size + self.extraBoard.shape[1], height * block_size))
        pygame.display.set_caption('Tetris')
        self.font = pygame.font.SysFont('arial', 18)
        self.reset()

    def getNextState(self):
        states = {}
        currPiece = self.piece[:]
        rotations = [0, 1] if len(self.piece) == len(self.piece[0]) else [0, 1, 2, 3]

        for rotation in rotations:
            currPiece = self.rotatePiece(currPiece)
            validXs = self.width - len(currPiece[0])
            for x in range(validXs + 1):
                pos = {"x": x, "y": 0}
                while not self.collisionDetection(currPiece, pos):
                    pos["y"] += 1
                board = self.store(currPiece, pos)
                states[(x, rotation)] = self.getStateProperties(board)
        return states

    def store(self, piece, pos):
        board = [row[:] for row in self.board]
        for y, row in enumerate(piece):
            for x, val in enumerate(row):
                if val and not board[pos["y"] + y][pos["x"] + x]:
                    board[pos["y"] + y][pos["x"] + x] = val
        return board

    def clearLines(self, board):
        toDelete = [i for i, row in enumerate(board) if all(row)]
        for i in reversed(toDelete):
            board.pop(i)
            board.insert(0, [0] * self.width)
        return len(toDelete), board

    def step(self, action, render=True):
        x, rotations = action
        self.currentPos = {"x": x, "y": 0}
        for _ in range(rotations):
            self.piece = self.rotatePiece(self.piece)

        while not self.collisionDetection(self.piece, self.currentPos):
            self.currentPos["y"] += 1
            if render:
                self.render()

        overflow = self.truncate(self.piece, self.currentPos)
        if overflow:
            self.gameOver = True

        self.board = self.store(self.piece, self.currentPos)

        linesCleared, self.board = self.clearLines(self.board)
        score = 1 + (linesCleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.clearedLines += linesCleared
        if not self.gameOver:
            self.newPiece()
        if self.gameOver:
            self.score -= 2

        return score, self.gameOver

    def render(self):
        self.screen.fill((0, 0, 0))
        for y, row in enumerate(self.board):
            for x, val in enumerate(row):
                color = self.pieceColors[val]
                pygame.draw.rect(self.screen, color, pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size))

        for y, row in enumerate(self.piece):
            for x, val in enumerate(row):
                if val:
                    color = self.pieceColors[val]
                    pygame.draw.rect(self.screen, color, pygame.Rect((self.currentPos["x"] + x) * self.block_size, (self.currentPos["y"] + y) * self.block_size, self.block_size, self.block_size))

        pygame.draw.rect(self.screen, (255, 255, 255), (self.width * self.block_size, 0, self.extraBoard.shape[1], self.extraBoard.shape[0]))
        pygame.draw.rect(self.screen, (0, 0, 0), (self.width * self.block_size, 0, self.extraBoard.shape[1], self.extraBoard.shape[0]), 1)

        self.screen.blit(self.font.render(f'Score: {self.score}', True, self.text_color), (self.width * self.block_size + 10, 20))
        self.screen.blit(self.font.render(f'Pieces: {self.tetrominoes}', True, self.text_color), (self.width * self.block_size + 10, 60))
        self.screen.blit(self.font.render(f'Lines: {self.clearedLines}', True, self.text_color), (self.width * self.block_size + 10, 100))

        pygame.display.flip()

def run():
    model_path = "trained_models/tetris_final"
    model = torch.load(model_path)
    model.eval()

    tetris = Tetris()
    clock = pygame.time.Clock()
    state = tetris.reset()

    while not tetris.gameOver:
        nextSteps = tetris.getNextState()
        nextActions, nextStates = zip(*nextSteps.items())
        nextStates = torch.stack(nextStates)
        with torch.no_grad():
            predictions = model(nextStates)[:, 0]
        index = torch.argmax(predictions).item()
        action = nextActions[index]
        tetris.step(action)
        tetris.render()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    run()
