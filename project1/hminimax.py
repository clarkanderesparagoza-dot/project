from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    """ตัวแทน Pacman ที่ใช้อัลกอริทึม Heuristic Minimax (H-Minimax)"""

    def __init__(self, depth=2):
        """
        Arguments:
            depth: ความลึกสูงสุดในการค้นหาของต้นไม้ Minimax
        """
        super().__init__()
        self.depth = int(depth)
        self.history = []  # บันทึกประวัติตำแหน่งที่ Pacman เดินผ่าน

    def evaluation_function(self, state):
        """ประเมินสถานะของเกมเมื่อการค้นหาถึงขีดจำกัดความลึก (Depth Limit)

        Arguments:
            state: สถานะปัจจุบันของเกม Pacman

        Returns:
            คะแนนการประเมินสถานะ (Heuristic Score)
        """
        if state.isWin():
            return 999999
        if state.isLose():
            return -999999

        score = state.getScore()
        pacman_pos = state.getPacmanPosition()
        food_list = state.getFood().asList()
        ghost_positions = state.getGhostPositions()

        # 1. เน้นกินอาหารให้หมด
        score -= 100 * len(food_list)

        # 2. คำนวณระยะทางไปยังอาหารที่ใกล้ที่สุด
        if food_list:
            min_food_dist = min(manhattanDistance(pacman_pos, f) for f in food_list)
            score -= 1.5 * min_food_dist

        # 3. คำนวณระยะห่างจากผี
        for ghost_pos in ghost_positions:
            ghost_dist = manhattanDistance(pacman_pos, ghost_pos)
            if ghost_dist <= 1:
                score -= 1000
            elif ghost_dist <= 2:
                score -= 300
            elif ghost_dist <= 3:
                score -= 50

        # 4. หักคะแนนเมื่อเดินซ้ำตำแหน่งเดิม
        if pacman_pos in self.history:
            recency = len(self.history) - self.history.index(pacman_pos)
            score -= 200 / recency

        return score

    def get_action(self, state):
        """รับสถานะปัจจุบันของเกม Pacman แล้วส่งคืนการเคลื่อนไหวที่ถูกกฎหมายโดยใช้ H-Minimax

        Arguments:
            state: สถานะปัจจุบันของเกม Pacman

        Returns:
            การเคลื่อนที่ที่ถูกกฎหมายตามที่นิยามใน game.Directions
        """
        def h_minimax(current_state, depth, agent_index, alpha, beta):
            """ค้นหาแบบเรียกตัวเองด้วย H-Minimax ร่วมกับ Alpha-Beta Pruning"""
            if current_state.isWin() or current_state.isLose() or depth == self.depth:
                return self.evaluation_function(current_state)

            num_agents = current_state.getNumAgents()
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth + 1 if next_agent == 0 else depth

            # เรียกใช้ API ขยายโหนดตามข้อกำหนดของโจทย์
            if agent_index == 0:
                successors = current_state.generatePacmanSuccessors()
            else:
                successors = current_state.generateGhostSuccessors(agent_index)

            # กรองการหยุดนิ่ง (Directions.STOP) ออก
            valid_successors = [(s, a) for s, a in successors if a != Directions.STOP]

            if not valid_successors:
                return self.evaluation_function(current_state)

            # โหนด MAX (Pacman)
            if agent_index == 0:
                best_score = float('-inf')
                for successor, action in valid_successors:
                    score = h_minimax(successor, next_depth, next_agent, alpha, beta)
                    best_score = max(best_score, score)
                    if best_score >= beta:
                        return best_score
                    alpha = max(alpha, best_score)
                return best_score

            # โหนด MIN (ผี)
            else:
                best_score = float('inf')
                for successor, action in valid_successors:
                    score = h_minimax(successor, next_depth, next_agent, alpha, beta)
                    best_score = min(best_score, score)
                    if best_score <= alpha:
                        return best_score
                    beta = min(beta, best_score)
                return best_score

        best_score = float('-inf')
        best_action = Directions.STOP
        alpha = float('-inf')
        beta = float('inf')

        # ขยายโหนดเริ่มต้นสำหรับ Pacman
        successors = state.generatePacmanSuccessors()
        valid_successors = [(s, a) for s, a in successors if a != Directions.STOP]

        for successor, action in valid_successors:
            score = h_minimax(successor, 0, 1, alpha, beta)

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        # อัปเดตประวัติการเดิน
        self.history.append(state.getPacmanPosition())
        if len(self.history) > 5:
            self.history.pop(0)

        return best_action