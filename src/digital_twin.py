# src/digital_twin.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

@dataclass
class SimulationResult:
    action: str
    production: float
    downtime: float
    risk: float
    cost: float
    explanation: str

class FactorySimulator:
    def __init__(self, initial_health=None, initial_wear=None, base_production_rate=100.0):
        self.machines = ['M1','M2','M3','M4']
        self.base_rate = base_production_rate
        self.health = initial_health or {m: 0.9 for m in self.machines}
        self.wear = initial_wear or {m: 0.2 for m in self.machines}
        self.running = {m: True for m in self.machines}

    def step(self, action, load_factor=1.0):
        if action == 'stop':
            for m in self.machines:
                self.running[m] = False
                self.wear[m] = max(0, self.wear[m] - 0.1)
                self.health[m] = min(1, self.health[m] + 0.05)
        elif action == 'reduce_load':
            load_factor = 0.7
        else:
            load_factor = 1.0

        total_prod = 0
        downtime = 0
        for m in self.machines:
            if self.running[m]:
                effective_rate = self.base_rate * self.health[m] * (1 - self.wear[m]) * load_factor
                noise = np.random.normal(1, 0.05)
                produced = max(0, effective_rate * noise)
                total_prod += produced
                self.wear[m] = min(1, self.wear[m] + 0.001 * produced / self.base_rate)
                self.health[m] = max(0, self.health[m] - 0.0005 * produced / self.base_rate)
            else:
                downtime += 1

        risk = max(0, 1 - min(self.health.values())) * 0.5 + max(0, max(self.wear.values()) - 0.7) * 0.5
        risk = min(1, risk)
        defect_rate = 0.05 + (1 - np.mean(list(self.health.values()))) * 0.3
        defect_cost = total_prod * defect_rate * 2
        downtime_cost = downtime * 100
        maintenance_cost = sum(self.wear.values()) * 50
        total_cost = downtime_cost + maintenance_cost + defect_cost

        return {'production': total_prod, 'downtime': downtime, 'risk': risk, 'cost': total_cost}

    def simulate(self, action, hours=24, load_factor=1.0):
        total_prod = 0
        total_downtime = 0
        max_risk = 0
        total_cost = 0
        for _ in range(hours):
            res = self.step(action, load_factor)
            total_prod += res['production']
            total_downtime += res['downtime']
            max_risk = max(max_risk, res['risk'])
            total_cost += res['cost']
        avg_cost = total_cost / hours
        explanation = (f"Action '{action}': Production {total_prod:.1f} units, "
                       f"Downtime {total_downtime:.1f}h, Risk {max_risk:.2f}, Cost {avg_cost:.1f}/h")
        return SimulationResult(action, total_prod, total_downtime, max_risk, avg_cost, explanation)

def compare_scenarios(current_state=None, hours=24):
    """
    Compare three scenarios: continue, stop, reduce_load.
    Each scenario starts with a fresh simulator (same initial state).
    """
    results = []
    for action in ['continue', 'stop', 'reduce_load']:
        # Create a new simulator for each scenario to avoid state contamination
        sim = FactorySimulator(
            initial_health=current_state.get('health') if current_state else None,
            initial_wear=current_state.get('wear') if current_state else None
        )
        res = sim.simulate(action, hours)
        results.append({
            'Scenario': action,
            'Production': round(res.production, 1),
            'Downtime (hrs)': round(res.downtime, 1),
            'Risk': round(res.risk, 3),
            'Cost (per hr)': round(res.cost, 1),
            'Explanation': res.explanation
        })
    return pd.DataFrame(results)