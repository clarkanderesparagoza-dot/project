# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module import util
import numpy as np
import random

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

        pacman_pos = state.getPacmanPosition()
        target_pos = None
        for b_state in belief_state:
            if np.sum(b_state) > 0: # ผีตัวนี้ยังไม่โดนกิน
                # ดึงพิกัด (x, y) ที่มีค่าความน่าจะเป็นสูงสุด
                target_pos = np.unravel_index(np.argmax(b_state), b_state.shape)
                break # ไล่ล่าทีละตัว

        if target_pos is None:
            # ถ้ากินผีหมดแล้ว หรือไม่มีข้อมูล ให้เดินแบบสุ่ม
            return random.choice(legal_moves)

        # 2. เลือกทิศทางที่ลดระยะทางไปหาเป้าหมายให้เหลือน้อยที่สุด (Greedy)
        best_move = Directions.STOP
        min_dist = float('inf')

        for move in legal_moves:
            next_pos = pacman_pos
            if move == Directions.NORTH:   next_pos = (pacman_pos[0], pacman_pos[1] + 1)
            elif move == Directions.SOUTH: next_pos = (pacman_pos[0], pacman_pos[1] - 1)
            elif move == Directions.EAST:  next_pos = (pacman_pos[0] + 1, pacman_pos[1])
            elif move == Directions.WEST:  next_pos = (pacman_pos[0] - 1, pacman_pos[1])

            dist = util.manhattanDistance(next_pos, target_pos)
            
            if dist < min_dist:
                min_dist = dist
                best_move = move

        return best_move

        # XXX: End of your code here to obtain bonus
        

        return Directions.STOP
