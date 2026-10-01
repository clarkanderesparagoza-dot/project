from collections import deque
from pacman_module.game import Agent, Directions


def key(state):
    """ส่งคืนคีย์ที่ระบุสถานะของเกม Pacman ได้อย่างไม่ซ้ำกัน

    Arguments:
        state: สถานะปัจจุบันของเกม Pacman

    Returns:
        ทูเพิลที่เป็นคีย์สำหรับระบุสถานะเกม
    """
    return (state.getPacmanPosition(), state.getFood())


class PacmanAgent(Agent):
    """ตัวแทน Pacman ที่ใช้อัลกอริทึมการค้นหาตามแนวกว้าง (Breadth-First Search)"""

    def __init__(self, args=None):
        """
        Arguments:
            args: อาร์กิวเมนต์จากคำสั่งในเทอร์มินัล
        """
        super().__init__()
        self.moves = []

    def get_action(self, state):
        """รับสถานะปัจจุบันของเกม Pacman แล้วส่งคืนการเคลื่อนไหวที่ถูกกฎหมายโดยใช้ BFS

        Arguments:
            state: สถานะปัจจุบันของเกม Pacman

        Returns:
            การเคลื่อนที่ที่ถูกกฎหมายตามที่นิยามใน game.Directions
        """
        if not self.moves:
            self.moves = self._bfs(state)

        if self.moves:
            return self.moves.pop(0)

        return Directions.STOP

    def _bfs(self, initial_state):
        """ค้นหาเส้นทางไปยังเป้าหมายด้วยอัลกอริทึม BFS

        Arguments:
            initial_state: สถานะเริ่มต้นของเกม Pacman

        Returns:
            รายการคำสั่งการเคลื่อนที่ (Actions) ที่นำไปสู่ชัยชนะ
        """
        frontier = deque([(initial_state, [])])
        visited = {key(initial_state)}

        while frontier:
            current_state, actions = frontier.popleft()

            if current_state.isWin():
                return actions

            # เรียกใช้ API ขยายโหนดสำหรับ Pacman ตามข้อกำหนด
            for successor, action in current_state.generatePacmanSuccessors():
                if action == Directions.STOP:
                    continue

                state_key = key(successor)
                if state_key not in visited:
                    visited.add(state_key)
                    frontier.append((successor, actions + [action]))

        return []