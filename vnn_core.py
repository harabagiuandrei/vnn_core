import numpy as np
from typing import List, Tuple, Dict, Optional
import uuid
import random
import math

# -----------------------------------------
# Force Resonance Profile – like TUF mapping
# -----------------------------------------
class ForceResonanceProfile:
    def __init__(self, em=0.0, gravity=0.0, quantum=0.0, psi=0.0):
        self.em = em              # Electromagnetic resonance
        self.gravity = gravity    # Gravitational response
        self.quantum = quantum    # Quantum pairing resonance
        self.psi = psi            # Informational / logical resonance

    def total_resonance(self):
        return self.em + self.gravity + self.quantum + self.psi

    def __str__(self):
        return f"R[EM:{self.em:.2f}, G:{self.gravity:.2f}, Q:{self.quantum:.2f}, Ψ:{self.psi:.2f}]"


# -----------------------------------------
# 12D Matrix Position Container
# -----------------------------------------
class VMatrix12D:
    def __init__(self, dimensions: Optional[int] = 12):
        self.position = np.random.rand(dimensions)  # 12D point

    def distance_to(self, other: 'VMatrix12D') -> float:
        return np.linalg.norm(self.position - other.position)

    def __str__(self):
        return f"Pos12D({', '.join(f'{x:.2f}' for x in self.position)})"


# -----------------------------------------
# Quantum Twin – Dual (or more) state logic
# -----------------------------------------
class QuantumTwin:
    def __init__(self, classical_state: float):
        self.classical = classical_state
        self.quantum = random.uniform(0.0, 1.0)
        self.entangled_with: Optional[uuid.UUID] = None

    def collapse(self, method='probabilistic'):
        if method == 'probabilistic':
            return self.quantum if random.random() < 0.5 else self.classical
        elif method == 'mean':
            return (self.quantum + self.classical) / 2

    def __str__(self):
        return f"QT[classic:{self.classical:.2f}, quantum:{self.quantum:.2f}]"


# -----------------------------------------
# Neuron (VNeuron) – Has memory, state, resonance
# -----------------------------------------
class VNeuron:
    def __init__(self):
        self.id = uuid.uuid4()
        self.position = VMatrix12D()
        self.resonance = ForceResonanceProfile(
            em=random.uniform(0, 1),
            gravity=random.uniform(0, 1),
            quantum=random.uniform(0, 1),
            psi=random.uniform(0, 1)
        )
        self.qstate = QuantumTwin(classical_state=random.uniform(0, 1))
        self.state = "dormant"  # or "active", "bound"
        self.memory: List[str] = []

    def activate(self):
        if self.state == "dormant":
            self.state = "active"
            self.memory.append(f"Activated at resonance {self.resonance}")
    
    def decay(self):
        if self.state == "active":
            self.state = "dormant"
            self.memory.append("Decayed")

    def bind(self, other: 'VNeuron'):
        # Bind neurons through logic/memory pairing
        self.state = "bound"
        other.state = "bound"
        self.memory.append(f"Bound with {other.id}")
        other.memory.append(f"Bound with {self.id}")

    def __str__(self):
        return f"Neuron[{self.id}] at {self.position} | State: {self.state} | {self.resonance}"


# -----------------------------------------
# RodGate – Forms a triangle between neurons
# -----------------------------------------
class VRodGate:
    def __init__(self, a: VNeuron, b: VNeuron, c: VNeuron):
        self.id = uuid.uuid4()
        self.neurons = (a, b, c)
        self.last_energy_flux: float = 0.0

    def evaluate_resonance(self):
        total = sum(n.resonance.total_resonance() for n in self.neurons)
        self.last_energy_flux = total / 3.0
        return self.last_energy_flux

    def trigger(self):
        avg_flux = self.evaluate_resonance()
        if avg_flux > 2.0:  # Arbitrary threshold
            for n in self.neurons:
                n.activate()

    def __str__(self):
        return f"Rod[{self.id}] linking {[n.id for n in self.neurons]} | Flux: {self.last_energy_flux:.2f}"


# -----------------------------------------
# VNN Core – Controls all logic, time steps, training
# -----------------------------------------
class VNNCore:
    def __init__(self, neuron_count: int = 100):
        self.neurons: List[VNeuron] = [VNeuron() for _ in range(neuron_count)]
        self.rodgates: List[VRodGate] = []
        self.time_step = 0
        self.generate_rods()

    def generate_rods(self):
        for _ in range(len(self.neurons) // 3):
            triplet = random.sample(self.neurons, 3)
            rod = VRodGate(*triplet)
            self.rodgates.append(rod)

    def evolve(self, steps=1):
        for _ in range(steps):
            self.time_step += 1
            for rod in self.rodgates:
                rod.trigger()
            self.adapt()

    def adapt(self):
        # Neurons can shift positions slightly, or rebind
        for neuron in self.neurons:
            if neuron.state == "active" and random.random() < 0.2:
                neuron.decay()
            elif neuron.state == "dormant" and random.random() < 0.05:
                neuron.activate()

    def status_report(self):
        active = len([n for n in self.neurons if n.state == "active"])
        dormant = len([n for n in self.neurons if n.state == "dormant"])
        bound = len([n for n in self.neurons if n.state == "bound"])
        return f"[t={self.time_step}] Active:{active} Dormant:{dormant} Bound:{bound}"

    def __str__(self):
        return f"VNNCore | Neurons: {len(self.neurons)} | Rods: {len(self.rodgates)} | Time: {self.time_step}"
