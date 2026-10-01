from pacman_module.game import Agent, Directions


def key(state):
    """ส่งคืนคีย์ที่ระบุสถานะของเกม Pacman ได้อย่างไม่ซ้ำกัน

    Arguments:
        state: สถานะปัจจุบันของเกม Pacman

    Returns:
        ทูเพิลที่เป็นคีย์สำหรับระบุสถานะเกม
    """
    pacman_pos = state.getPacmanPosition()
    ghost_positions = tuple(state.getGhostPositions())
    food_grid = state.getFood()
    return (pacman_pos, ghost_positions, food_grid)


class PacmanAgent(Agent):
    """ตัวแทน Pacman ที่ใช้อัลกอริทึม Minimax ร่วมกับ Alpha-Beta Pruning"""

    def __init__(self, args=None):
        """
        Arguments:
            args: อาร์กิวเมนต์จากคำสั่งในเทอร์มินัล
        """
        super().__init__()

    def get_action(self, state):
        """รับสถานะปัจจุบันของเกม Pacman แล้วส่งคืนการเคลื่อนไหวที่ถูกกฎหมายโดยใช้ Minimax

        Arguments:
            state: สถานะปัจจุบันของเกม Pacman

        Returns:
            การเคลื่อนที่ที่ถูกกฎหมายตามที่นิยามใน game.Directions
        """
        def minimax(current_state, agent_index, path, alpha, beta):
            """ค้นหาแบบเรียกตัวเองด้วย Minimax ร่วมกับ Alpha-Beta Pruning และการตรวจจับลูป"""
            if current_state.isWin() or current_state.isLose():
                return current_state.getScore()

            state_key = key(current_state)
            if state_key in path:
                return -99999 if agent_index == 0 else 99999

            new_path = path | {state_key}
            num_agents = current_state.getNumAgents()
            next_agent = (agent_index + 1) % num_agents

            # เรียกใช้ API ขยายโหนดตามข้อกำหนดของโจทย์
            if agent_index == 0:
                successors = current_state.generatePacmanSuccessors()
            else:
                successors = current_state.generateGhostSuccessors(agent_index)

            # กรองการหยุดนิ่ง (Directions.STOP) ออก
            valid_successors = [(s, a) for s, a in successors if a != Directions.STOP]

            if not valid_successors:
                return current_state.getScore()

            # โหนด MAX (Pacman)
            if agent_index == 0:
                best_score = float('-inf')
                for successor, action in valid_successors:
                    score = minimax(successor, next_agent, new_path, alpha, beta)
                    best_score = max(best_score, score)
                    if best_score >= beta:
                        return best_score
                    alpha = max(alpha, best_score)
                return best_score

            # โหนด MIN (ผี)
            else:
                best_score = float('inf')
                for successor, action in valid_successors:
                    score = minimax(successor, next_agent, new_path, alpha, beta)
                    best_score = min(best_score, score)
                    if best_score <= alpha:
                        return best_score
                    beta = min(beta, best_score)
                return best_score

        best_score = float('-inf')
        best_action = Directions.STOP
        alpha = float('-inf')
        beta = float('inf')
        initial_path = frozenset([key(state)])

        # ขยายโหนดเริ่มต้นสำหรับ Pacman
        successors = state.generatePacmanSuccessors()
        valid_successors = [(s, a) for s, a in successors if a != Directions.STOP]

        for successor, action in valid_successors:
            score = minimax(successor, 1, initial_path, alpha, beta)

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        return best_action