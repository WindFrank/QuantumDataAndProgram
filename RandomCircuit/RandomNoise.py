import cirq
import random


class RandomNoise(cirq.Gate):
    def __init__(self, all_p=0.5) -> None:
        self._p = 0
        self.random1 = 0
        self.random2 = 0
        self.random3 = 0
        while self.random1 + self.random2 == 0 or self.random1 + self.random2 >= all_p:
            self.random1 = random.random()
            self.random2 = random.random()
            self.random3 = all_p - self.random1 - self.random2

    def _num_qubits_(self):
        return 1

    def _mixture_(self):
        ps = [1.0 - self.random1 - self.random2 - self.random3, self.random1, self.random2, self.random3]
        ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.X), cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        return True

    def set_concrete_p(self, random1, random2, random3):
        all_p = random1 + random2 + random3
        self.random1 = random1
        self.random2 = random2
        self.random3 = random3

    def _circuit_diagram_info_(self, args) -> str:
        return f"RandomNoise({self.random1}, {self.random2}, {self.random3})"



