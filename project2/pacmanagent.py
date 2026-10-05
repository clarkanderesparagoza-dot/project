# Complete this class for all parts of the project

from collections import deque

from pacman_module.game import Agent, Actions
from pacman_module.pacman import Directions
import numpy as np
import random


def bfs_distances(walls, start):
    """Shortest path lengths (in moves) from a cell to every other cell.

    Arguments:
    ----------
    - `walls`: the grid of walls, as given by `state.getWalls()`.
    - `start`: the (x, y) cell to start from.

    Return:
    -------
    - A 2D array with the number of moves needed to reach each cell.
      Walls and unreachable cells hold `width * height`.
    """
    unreachable = walls.width * walls.height
    dist = np.full((walls.width, walls.height), unreachable, dtype=float)
    dist[start[0], start[1]] = 0
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < walls.width and 0 <= ny < walls.height):
                continue
            if walls[nx][ny] or dist[nx, ny] != unreachable:
                continue
            dist[nx, ny] = dist[x, y] + 1
            queue.append((nx, ny))
    return dist


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
                returns a legal move.

        Pacman chases the ghost that is the closest on average (according
        to its belief) and takes the move that minimizes the expected
        shortest-path distance (walls included) to this ghost.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        # XXX: Your code here to obtain bonus
        legal_moves = state.getLegalPacmanActions()
        if Directions.STOP in legal_moves:
            legal_moves.remove(Directions.STOP)

        if not legal_moves:
            return Directions.STOP

        # A ghost that has been eaten has a belief of zeros
        alive = [b for b in belief_state if np.sum(b) > 0]
        if not alive:
            return random.choice(legal_moves)

        walls = state.getWalls()
        x, y = state.getPacmanPosition()
        pacman_pos = (int(x), int(y))

        # Target: the ghost that is the closest on average
        dist_here = bfs_distances(walls, pacman_pos)
        target = min(alive, key=lambda b: np.sum(b * dist_here))

        # Move: the one that brings Pacman closest to the belief of the target
        best_move = legal_moves[0]
        best_score = float('inf')
        for move in legal_moves:
            dx, dy = Actions.directionToVector(move)
            next_pos = (pacman_pos[0] + int(dx), pacman_pos[1] + int(dy))
            score = np.sum(target * bfs_distances(walls, next_pos))
            if score < best_score - 1e-12:
                best_score = score
                best_move = move

        return best_move
        # XXX: End of your code here to obtain bonus