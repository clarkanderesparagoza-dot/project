from pacman_module.game import Agent
import numpy as np
from pacman_module import util
from scipy.stats import binom


class BeliefStateAgent(Agent):
    """ตัวแทน Pacman ที่ใช้ Bayes Filter ในการคาดเดาตำแหน่งของผี"""

    def __init__(self, args):
        """
        Arguments:
            args: อาร์กิวเมนต์จากคำสั่งในเทอร์มินัล
        """
        super().__init__()
        self.args = args

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        self.p = 0.5
        self.n = int(self.sensor_variance / (self.p * (1 - self.p)))

        # กำหนดพารามิเตอร์พฤติกรรมความกลัวของผี (p_flee) สำหรับ Transition Model
        if self.ghost_type == "scared":
            self.p_flee = 0.8
        elif self.ghost_type == "afraid":
            self.p_flee = 0.5
        elif self.ghost_type == "confused":
            self.p_flee = 0.2
        else:
            self.p_flee = 0.5

        # ตัวแปรสำหรับบันทึกข้อมูล Metrics นำไปพล็อตกราฟรายงาน
        self.metrics_history = []

    def _get_sensor_model(self, pacman_position, evidence):
        """คำนวณแบบจำลองเซนเซอร์ P(E_t = evidence | X_t = (w, h))

        Arguments:
            pacman_position: ตำแหน่งพิกัด (x, y) ของ Pacman ที่เวลา t
            evidence: ระยะทางจากเซนเซอร์ที่มีสัญญาณรบกวน (Noisy distance)

        Returns:
            อาร์เรย์ NumPy 2D ขนาด [width, height] แสดงความน่าจะเป็น
        """
        width = self.walls.width
        height = self.walls.height
        sensor_model = np.zeros((width, height))

        for w in range(width):
            for h in range(height):
                if not self.walls[w][h]:
                    true_dist = util.manhattanDistance((w, h), pacman_position)
                    B = (evidence - true_dist) + (self.n * self.p)
                    if 0 <= B <= self.n and abs(B - round(B)) < 1e-5:
                        prob = binom.pmf(round(B), self.n, self.p)
                        sensor_model[w, h] = prob

        return sensor_model

    def _get_transition_model(self, pacman_position):
        """คำนวณแบบจำลองการเปลี่ยนสถานะ P(X_{t+1} = (w1, h1) | X_t = (w2, h2))

        Arguments:
            pacman_position: ตำแหน่งพิกัด (x, y) ของ Pacman ที่เวลา t

        Returns:
            อาร์เรย์ NumPy 4D ขนาด [width, height, width, height]
        """
        width = self.walls.width
        height = self.walls.height
        transition_model = np.zeros((width, height, width, height))

        for w2 in range(width):
            for h2 in range(height):
                if not self.walls[w2][h2]:
                    # ค้นหาตำแหน่งเพื่อนบ้านที่เดินไปได้ (Legal moves)
                    legal_moves = []
                    for w1, h1 in [
                        (w2, h2 + 1),
                        (w2, h2 - 1),
                        (w2 + 1, h2),
                        (w2 - 1, h2),
                    ]:
                        if (
                            0 <= w1 < width
                            and 0 <= h1 < height
                            and not self.walls[w1][h1]
                        ):
                            legal_moves.append((w1, h1))

                    if legal_moves:
                        # คำนวณความต่างของระยะทางเพื่อดูว่าการเดินทิศทางไหนเป็นการหนี Pacman
                        curr_dist = util.manhattanDistance((w2, h2), pacman_position)
                        flee_moves = []
                        other_moves = []

                        for w1, h1 in legal_moves:
                            next_dist = util.manhattanDistance((w1, h1), pacman_position)
                            if next_dist > curr_dist:
                                flee_moves.append((w1, h1))
                            else:
                                other_moves.append((w1, h1))

                        # ให้ค่าน้ำหนักความน่าจะเป็นตามพารามิเตอร์ p_flee ของผี
                        if flee_moves and other_moves:
                            prob_flee = self.p_flee / len(flee_moves)
                            prob_other = (1.0 - self.p_flee) / len(other_moves)
                            for w1, h1 in flee_moves:
                                transition_model[w1, h1, w2, h2] = prob_flee
                            for w1, h1 in other_moves:
                                transition_model[w1, h1, w2, h2] = prob_other
                        else:
                            # กรณีไม่มีทางเลือกหนี ให้แบ่งความน่าจะเป็นเท่ากัน (Uniform)
                            prob = 1.0 / len(legal_moves)
                            for w1, h1 in legal_moves:
                                transition_model[w1, h1, w2, h2] = prob

        return transition_model

    def _get_updated_belief(
        self, belief, evidences, pacman_position, ghosts_eaten
    ):
        """คำนวณอัปเดต Belief State ของผีแต่ละตัวที่เวลา t

        Arguments:
            belief: รายการของ Belief States ในช่วงเวลาก่อนหน้า (t-1)
            evidences: รายการข้อมูลระยะทางที่มีสัญญาณรบกวน
            pacman_position: ตำแหน่งพิกัด Pacman ปัจจุบัน
            ghosts_eaten: สถานะการโดนกินของผีแต่ละตัว (Boolean)

        Returns:
            รายการของ Belief States อัปเดตล่าสุดที่เวลา t
        """
        new_belief = []
        width = self.walls.width
        height = self.walls.height

        trans_model = self._get_transition_model(pacman_position)

        for z in range(len(belief)):
            if ghosts_eaten[z]:
                # หากผีโดนกินแล้ว ปรับความน่าจะเป็นเป็น 0 ทั้งแผนที่
                new_belief.append(np.zeros((width, height)))
            else:
                # 1. Prediction Step (Time Update)
                predicted_belief = np.zeros((width, height))
                for w1 in range(width):
                    for h1 in range(height):
                        predicted_belief[w1, h1] = np.sum(
                            trans_model[w1, h1, :, :] * belief[z]
                        )

                # 2. Measurement Update Step
                sensor_model = self._get_sensor_model(
                    pacman_position, evidences[z]
                )
                updated_belief = sensor_model * predicted_belief

                # 3. Normalization Step
                total_prob = np.sum(updated_belief)
                if total_prob > 0:
                    updated_belief /= total_prob
                else:
                    # Fallback กรณีไม่พบผี ปรับให้กระจายความน่าจะเป็นแบบสม่ำเสมอ
                    updated_belief = np.ones((width, height)) / (width * height)

                new_belief.append(updated_belief)

        return new_belief

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """ห้ามแก้ไขฟังก์ชันนี้ ตามข้อกำหนดโปรเจกต์"""
        belief = self._get_updated_belief(
            self.beliefGhostStates, evidences, pacman_position, ghosts_eaten
        )
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """ห้ามแก้ไขฟังก์ชันนี้ ตามข้อกำหนดโปรเจกต์"""
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n * self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """คำนวณและบันทึกค่า Metrics สำหรับรายงาน (คำตอบข้อ 3.a และ 3.b)

        Arguments:
            belief_states: รายการ Belief States ปัจจุบัน
            state: สถานะเกมปัจจุบัน (ใช้สำหรับดึง True Positions)
        """
        true_positions = state.getGhostPositions()
        ghosts_eaten = state.data._eaten[1:]

        for z, b_state in enumerate(belief_states):
            if not ghosts_eaten[z]:
                # 3.a: ความไม่แน่นอน (Uncertainty / Entropy)
                flat_b = b_state.flatten()
                flat_b = flat_b[flat_b > 0]
                entropy = -np.sum(flat_b * np.log2(flat_b)) if len(flat_b) > 0 else 0

                # 3.b: ประสิทธิภาพการคาดการณ์ (Quality / Mean Absolute Error - MAE)
                gx, gy = int(true_positions[z][0]), int(true_positions[z][1])
                grid_x, grid_y = np.indices(b_state.shape)
                mae = np.sum(
                    b_state * (np.abs(grid_x - gx) + np.abs(grid_y - gy))
                )

                self.metrics_history.append((entropy, mae))

    def get_action(self, state):
        """ห้ามแก้ไขฟังก์ชันนี้ ตามข้อกำหนดโปรเจกต์"""
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(
            evidence, state.getPacmanPosition(), state.data._eaten[1:]
        )
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates, evidence