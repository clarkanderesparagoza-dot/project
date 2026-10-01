import heapq
from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance


def key(state):
    """ส่งคืนคีย์ที่ระบุสถานะของเกม Pacman ได้อย่างไม่ซ้ำกัน

    Arguments:
        state: สถานะปัจจุบันของเกม Pacman

    Returns:
        ทูเพิลที่เป็นคีย์สำหรับระบุสถานะเกม
    """
    return (state.getPacmanPosition(), state.getFood())


def heuristic(state):
    """คำนวณค่าประมาณการฮิวริสติกจากสถานะปัจจุบันไปยังเป้าหมาย

    Arguments:
        state: สถานะปัจจุบันของเกม Pacman

    Returns:
        ระยะทางแมนฮัตตันไปยังจุดอาหารที่ใกล้ที่สุด
    """
    pacman_pos = state.getPacmanPosition()
    food_list = state.getFood().asList()

    if not food_list:
        return 0

    return min(manhattanDistance(pacman_pos, food) for food in food_list)


class PacmanAgent(Agent):
    """ตัวแทน Pacman ที่ใช้อัลกอริทึมการค้นหาแบบเอสตาร์ (A* Search)"""

    def __init__(self, args=None):
        """
        Arguments:
            args: อาร์กิวเมนต์จากคำสั่งในเทอร์มินัล
        """
        super().__init__()
        self.moves = []

    def get_action(self, state):
        """รับสถานะปัจจุบันของเกม Pacman แล้วส่งคืนการเคลื่อนไหวที่ถูกกฎหมายโดยใช้ A*

        Arguments:
            state: สถานะปัจจุบันของเกม Pacman

        Returns:
            การเคลื่อนที่ที่ถูกกฎหมายตามที่นิยามใน game.Directions
        """
        if not self.moves:
            self.moves = self._astar(state)

        if self.moves:
            return self.moves.pop(0)

        return Directions.STOP

    def _astar(self, initial_state):
        """ค้นหาเส้นทางไปยังเป้าหมายด้วยอัลกอริทึม A* Search

        Arguments:
            initial_state: สถานะเริ่มต้นของเกม Pacman

        Returns:
            รายการคำสั่งการเคลื่อนที่ (Actions) ที่นำไปสู่ชัยชนะ
        """
        counter = 0
        frontier = []
        # เก็บข้อมูลแบบ (f_score, counter, current_state, actions, g_score)
        heapq.heappush(
            frontier,
            (heuristic(initial_state), counter, initial_state, [], 0)
        )

        cost_so_far = {key(initial_state): 0}

        while frontier:
            _, _, current_state, actions, g_score = heapq.heappop(frontier)

            if current_state.isWin():
                return actions

            current_key = key(current_state)
            if g_score > cost_so_far.get(current_key, float('inf')):
                continue

            # เรียกใช้ API ขยายโหนดสำหรับ Pacman ตามข้อกำหนด
            for successor, action in current_state.generatePacmanSuccessors():
                if action == Directions.STOP:
                    continue

                new_g = g_score + 1
                successor_key = key(successor)

                if successor_key not in cost_so_far or new_g < cost_so_far[successor_key]:
                    cost_so_far[successor_key] = new_g
                    counter += 1
                    f_score = new_g + heuristic(successor)
                    heapq.heappush(
                        frontier,
                        (f_score, counter, successor, actions + [action], new_g)
                    )

        return []