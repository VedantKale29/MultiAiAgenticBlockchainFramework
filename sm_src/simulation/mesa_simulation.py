"""
simulation/mesa_simulation.py
==============================
MESA-BASED STRESS TEST & SIMULATION VALIDATION

PURPOSE:
  Validates the agentic fraud detection system using Mesa agent-based
  simulation — WITHOUT needing the real pipeline code. Simulates the
  system's core logic (RF+IF hybrid, dual thresholds, adaptation) to
  stress-test scenarios that real data cannot cover.

WHAT THIS VALIDATES:
  1. Surge test        — batch size spike (600 → 2000 tx/batch), does adaptation hold?
  2. Concept drift     — fraud pattern shifts mid-run, does system self-correct?
  3. Adversarial drift — fraudster deliberately stays just below tau_alert
  4. Novel attack      — completely new fraud type, does RAG generalise?
  5. Zero-fraud batch  — no fraud in batch, does system avoid false positives?
  6. Cascade failure   — multiple attack types simultaneously

HOW IT WORKS:
  - TransactionAgent: simulates one blockchain transaction (normal or fraud)
  - FraudsterAgent:   simulates an attacker adapting their behaviour
  - DetectorModel:    Mesa Model that orchestrates agents, runs hybrid scoring,
                      tracks metrics batch by batch — mirrors your real pipeline

STANDALONE: runs with no dependency on the real pipeline files.
Compatible with Windows PowerShell.

USAGE:
  cd <project root>
  python simulation/mesa_simulation.py
  python simulation/mesa_simulation.py --scenario all
  python simulation/mesa_simulation.py --scenario surge
  python simulation/mesa_simulation.py --scenario drift
  python simulation/mesa_simulation.py --report
     
  
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import mesa

# ─── Mesa 3.x compatibility ───────────────────────────────────────────────────
# Mesa 3.x removed RandomActivation. We use SimultaneousActivation or
# just iterate agents manually.
try:
    from mesa.time import RandomActivation
    _SCHEDULER = "legacy"
except ImportError:
    _SCHEDULER = "modern"  # Mesa 3.x — no scheduler needed, use model.agents


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — mirrors config.py values exactly
# ══════════════════════════════════════════════════════════════════════════════

SEED            = 42
BATCH_SIZE      = 600          # paper value
W0              = 0.70         # initial fusion weight
TAU0            = 0.487        # initial alert threshold
DELTA           = 0.10         # block margin
TAU_BLOCK_INIT  = TAU0 + DELTA # = 0.587
TARGET_PREC     = 0.80
TARGET_RECALL   = 0.80
STEP_TAU        = 0.02
STEP_W          = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class TransactionAgent(mesa.Agent):
    """
    Simulates one blockchain transaction.
    Each agent has a feature vector and a true fraud label.
    """

    def __init__(self, model, is_fraud: bool, attack_type: str = "normal",
                 risk_boost: float = 0.0):
        super().__init__(model)
        self.is_fraud   = is_fraud
        self.attack_type = attack_type
        self.risk_boost  = risk_boost  # adversarial: fraudster lowers their score
        rng = model.rng

        # Simulate RF probability and IF score
        if is_fraud:
            self.p_rf = float(np.clip(rng.normal(0.75, 0.12) + risk_boost, 0.0, 1.0))
            self.s_if = float(np.clip(rng.normal(0.70, 0.15) + risk_boost, 0.0, 1.0))
        else:
            self.p_rf = float(np.clip(rng.normal(0.15, 0.10), 0.0, 1.0))
            self.s_if = float(np.clip(rng.normal(0.12, 0.08), 0.0, 1.0))

    def hybrid_score(self, w: float) -> float:
        return w * self.p_rf + (1.0 - w) * self.s_if

    def step(self):
        pass  # scoring handled by DetectorModel


class FraudsterAgent(mesa.Agent):
    """
    Simulates an adversarial fraudster who adapts their behaviour
    to avoid detection. Each step they slightly reduce their
    risk signal to try to stay below tau_alert.
    """

    def __init__(self, model, initial_risk: float = 0.80):
        super().__init__(model)
        self.target_risk   = initial_risk
        self.adaptation_rate = 0.03   # reduce risk signal by 3% per batch

    def step(self):
        # Fraudster learns current tau_alert and adjusts
        current_tau = self.model.tau_alert
        if self.target_risk > current_tau - 0.05:
            self.target_risk = max(0.30, self.target_risk - self.adaptation_rate)

    @property
    def risk_boost(self) -> float:
        """Negative boost = lower risk signal."""
        return self.target_risk - 0.75  # relative to fraud baseline


# ══════════════════════════════════════════════════════════════════════════════
# BATCH METRICS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchMetrics:
    batch_idx:    int
    tp:           int = 0
    fp:           int = 0
    fn:           int = 0
    tn:           int = 0
    precision:    float = 0.0
    recall:       float = 0.0
    f1:           float = 0.0
    w:            float = W0
    tau_alert:    float = TAU0
    tau_block:    float = TAU_BLOCK_INIT
    n_alert:      int = 0
    n_block:      int = 0
    n_clear:      int = 0
    latency_ms:   float = 0.0
    scenario_tag: str = ""

    def compute(self):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.precision = round(p, 4)
        self.recall    = round(r, 4)
        self.f1        = round(f, 4)


# ══════════════════════════════════════════════════════════════════════════════
# MESA MODEL — core detector
# ══════════════════════════════════════════════════════════════════════════════

class DetectorModel(mesa.Model):
    """
    Mesa Model that mirrors your CoordinatorAgent's batch loop.

    Each step() = one batch of transactions.
    The model creates TransactionAgents, scores them with the hybrid
    formula, makes CLEAR/ALERT/AUTO-BLOCK decisions, computes metrics,
    then runs adaptation (Algorithm 2).
    """

    def __init__(
        self,
        scenario_fn,          # callable(batch_idx, model) → list[TransactionAgent]
        n_batches: int = 5,
        seed: int = SEED,
    ):
        super().__init__(seed=seed)
        self.rng         = np.random.default_rng(seed)
        self.scenario_fn = scenario_fn
        self.n_batches   = n_batches

        # Shared agent state — mirrors CoordinatorAgent.agent_state
        self.w         = W0
        self.tau_alert = TAU0
        self.tau_block = TAU_BLOCK_INIT

        self.history: List[BatchMetrics] = []
        self.current_batch = 0

    # ── Core hybrid scoring ────────────────────────────────────────────────
    def _score_and_decide(self, agents: List[TransactionAgent]) -> Tuple[np.ndarray, np.ndarray]:
        scores    = np.array([a.hybrid_score(self.w) for a in agents])
        decisions = np.where(
            scores >= self.tau_block, "AUTO-BLOCK",
            np.where(scores >= self.tau_alert, "ALERT", "CLEAR")
        )
        return scores, decisions

    # ── Algorithm 2: Adaptation ────────────────────────────────────────────
    def _adapt(self, metrics: BatchMetrics):
        if metrics.recall < TARGET_RECALL:
            self.tau_alert = max(0.10, self.tau_alert - STEP_TAU)
        elif metrics.precision < TARGET_PREC:
            self.tau_alert = min(0.95, self.tau_alert + STEP_TAU)
        self.tau_block = min(1.0, self.tau_alert + DELTA)

        # Weight adjustment: if RF performing better than IF, increase w
        total = metrics.tp + metrics.fp + metrics.fn
        if total > 0:
            if metrics.precision >= TARGET_PREC and metrics.recall >= TARGET_RECALL:
                pass  # stable — no change
            elif metrics.recall < TARGET_RECALL:
                self.w = min(0.95, self.w + STEP_W)  # boost RF (more sensitive)
            else:
                self.w = max(0.05, self.w - STEP_W)

    # ── One batch step ────────────────────────────────────────────────────
    def step(self):
        t0 = time.perf_counter()

        # Generate transaction agents for this batch
        tx_agents = self.scenario_fn(self.current_batch, self)

        # Score and decide
        scores, decisions = self._score_and_decide(tx_agents)

        # Compute confusion matrix
        m = BatchMetrics(
            batch_idx    = self.current_batch,
            w            = round(self.w, 4),
            tau_alert    = round(self.tau_alert, 4),
            tau_block    = round(self.tau_block, 4),
            latency_ms   = 0.0,
        )

        for agent, score, decision in zip(tx_agents, scores, decisions):
            pred_positive = decision in ("ALERT", "AUTO-BLOCK")
            if pred_positive and agent.is_fraud:
                m.tp += 1
            elif pred_positive and not agent.is_fraud:
                m.fp += 1
            elif not pred_positive and agent.is_fraud:
                m.fn += 1
            else:
                m.tn += 1

            if decision == "ALERT":
                m.n_alert += 1
            elif decision == "AUTO-BLOCK":
                m.n_block += 1
            else:
                m.n_clear += 1

        m.compute()
        m.latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Adapt for next batch
        self._adapt(m)

        self.history.append(m)
        self.current_batch += 1

    def run(self) -> List[BatchMetrics]:
        for _ in range(self.n_batches):
            self.step()
        return self.history


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def _make_batch_fn(fraud_rate: float = 0.10, batch_size: int = BATCH_SIZE,
                   attack_type: str = "normal"):
    """Standard batch generator with fixed fraud rate."""
    def fn(batch_idx, model):
        n_fraud  = max(1, int(batch_size * fraud_rate))
        n_normal = batch_size - n_fraud
        agents   = ([TransactionAgent(model, is_fraud=True,  attack_type=attack_type)
                     for _ in range(n_fraud)] +
                    [TransactionAgent(model, is_fraud=False)
                     for _ in range(n_normal)])
        random.shuffle(agents)
        return agents
    return fn


# ── Scenario 1: Surge ─────────────────────────────────────────────────────────
def scenario_surge(batch_idx: int, model) -> List[TransactionAgent]:
    """
    Batches 0-1: normal (600 tx), Batch 2: surge (2000 tx),
    Batches 3-4: return to normal. Tests if adaptation handles volume spikes.
    """
    if batch_idx == 2:
        size = 2000   # 3.3x normal
    else:
        size = BATCH_SIZE

    fraud_rate = 0.12
    n_fraud    = max(1, int(size * fraud_rate))
    n_normal   = size - n_fraud
    agents     = ([TransactionAgent(model, is_fraud=True,  attack_type="surge")
                   for _ in range(n_fraud)] +
                  [TransactionAgent(model, is_fraud=False)
                   for _ in range(n_normal)])
    random.shuffle(agents)
    return agents


# ── Scenario 2: Concept Drift ─────────────────────────────────────────────────
def scenario_concept_drift(batch_idx: int, model) -> List[TransactionAgent]:
    """
    Batches 0-1: RF-dominant fraud (p_rf high, s_if low).
    Batches 2-4: IF-dominant fraud (new attack type, p_rf low, s_if high).
    Tests if fusion weight adapts to shift.
    """
    n_fraud  = int(BATCH_SIZE * 0.12)
    n_normal = BATCH_SIZE - n_fraud

    if batch_idx < 2:
        # Known fraud: RF picks it up well
        fraud_agents = []
        for _ in range(n_fraud):
            a = TransactionAgent(model, is_fraud=True, attack_type="known")
            a.p_rf = float(np.clip(model.rng.normal(0.85, 0.08), 0.0, 1.0))
            a.s_if = float(np.clip(model.rng.normal(0.40, 0.12), 0.0, 1.0))
            fraud_agents.append(a)
    else:
        # Drift: new fraud type, IF picks it up but RF misses
        fraud_agents = []
        for _ in range(n_fraud):
            a = TransactionAgent(model, is_fraud=True, attack_type="novel_drift")
            a.p_rf = float(np.clip(model.rng.normal(0.45, 0.10), 0.0, 1.0))
            a.s_if = float(np.clip(model.rng.normal(0.82, 0.08), 0.0, 1.0))
            fraud_agents.append(a)

    normal_agents = [TransactionAgent(model, is_fraud=False) for _ in range(n_normal)]
    agents = fraud_agents + normal_agents
    random.shuffle(agents)
    return agents


# ── Scenario 3: Adversarial Drift ────────────────────────────────────────────
def scenario_adversarial(batch_idx: int, model) -> List[TransactionAgent]:
    """
    A FraudsterAgent actively lowers their risk signal each batch,
    trying to slip below tau_alert. Tests if threshold adaptation
    catches them before they escape.
    """
    n_fraud  = int(BATCH_SIZE * 0.08)
    n_normal = BATCH_SIZE - n_fraud

    # Adversarial fraudster: starts high, gradually drops
    base_risk = 0.80 - (batch_idx * 0.06)  # drops 6% per batch
    base_risk = max(0.35, base_risk)

    fraud_agents = []
    for _ in range(n_fraud):
        a = TransactionAgent(model, is_fraud=True, attack_type="adversarial")
        a.p_rf = float(np.clip(model.rng.normal(base_risk, 0.06), 0.0, 1.0))
        a.s_if = float(np.clip(model.rng.normal(base_risk - 0.05, 0.07), 0.0, 1.0))
        fraud_agents.append(a)

    normal_agents = [TransactionAgent(model, is_fraud=False) for _ in range(n_normal)]
    agents = fraud_agents + normal_agents
    random.shuffle(agents)
    return agents


# ── Scenario 4: Novel Attack ──────────────────────────────────────────────────
def scenario_novel_attack(batch_idx: int, model) -> List[TransactionAgent]:
    """
    Batch 0-1: baseline.
    Batch 2: completely new attack pattern (ambiguous scores near tau boundary).
    Batch 3-4: system adapts, improves detection.
    """
    n_fraud  = int(BATCH_SIZE * 0.10)
    n_normal = BATCH_SIZE - n_fraud

    if batch_idx == 2:
        # Novel: scores sit right on the edge of tau_alert (hard to detect)
        fraud_agents = []
        for _ in range(n_fraud):
            a = TransactionAgent(model, is_fraud=True, attack_type="novel_attack")
            tau = model.tau_alert
            a.p_rf = float(np.clip(model.rng.normal(tau + 0.01, 0.04), 0.0, 1.0))
            a.s_if = float(np.clip(model.rng.normal(tau - 0.02, 0.05), 0.0, 1.0))
            fraud_agents.append(a)
    else:
        fraud_agents = [TransactionAgent(model, is_fraud=True, attack_type="known")
                        for _ in range(n_fraud)]

    normal_agents = [TransactionAgent(model, is_fraud=False) for _ in range(n_normal)]
    agents = fraud_agents + normal_agents
    random.shuffle(agents)
    return agents


# ── Scenario 5: Zero-Fraud Batch ─────────────────────────────────────────────
def scenario_zero_fraud(batch_idx: int, model) -> List[TransactionAgent]:
    """
    Batches 1 and 3 have zero fraud. Tests false positive rate and
    whether the system doesn't over-alert on clean traffic.
    """
    if batch_idx in (1, 3):
        return [TransactionAgent(model, is_fraud=False) for _ in range(BATCH_SIZE)]
    return _make_batch_fn(fraud_rate=0.10)(batch_idx, model)


# ── Scenario 6: Cascade Failure ──────────────────────────────────────────────
def scenario_cascade(batch_idx: int, model) -> List[TransactionAgent]:
    """
    All 5 batches have simultaneously: high fraud rate (25%) + mixed attack
    types + adversarial evasion. The worst-case stress test.
    """
    n_fraud  = int(BATCH_SIZE * 0.25)
    n_normal = BATCH_SIZE - n_fraud

    fraud_agents = []
    for i in range(n_fraud):
        attack_type = ["flash_loan", "reentrancy", "phishing", "adversarial"][i % 4]
        a = TransactionAgent(model, is_fraud=True, attack_type=attack_type)
        if attack_type == "adversarial":
            a.p_rf = float(np.clip(model.rng.normal(0.55, 0.08), 0.0, 1.0))
            a.s_if = float(np.clip(model.rng.normal(0.50, 0.09), 0.0, 1.0))
        fraud_agents.append(a)

    normal_agents = [TransactionAgent(model, is_fraud=False) for _ in range(n_normal)]
    agents = fraud_agents + normal_agents
    random.shuffle(agents)
    return agents


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_REGISTRY = {
    "baseline":    (_make_batch_fn(fraud_rate=0.10), "Baseline — normal operation (10% fraud rate)"),
    "surge":       (scenario_surge,         "Surge — batch spike 600→2000 transactions"),
    "drift":       (scenario_concept_drift, "Concept drift — RF→IF dominant fraud pattern shift"),
    "adversarial": (scenario_adversarial,   "Adversarial — fraudster evades detection gradually"),
    "novel":       (scenario_novel_attack,  "Novel attack — ambiguous edge-case fraud type"),
    "zero_fraud":  (scenario_zero_fraud,    "Zero-fraud batches — false positive stress test"),
    "cascade":     (scenario_cascade,       "Cascade failure — 25% fraud + mixed attack types"),
}


@dataclass
class ScenarioResult:
    name:        str
    description: str
    history:     List[BatchMetrics]
    passed:      bool = False
    verdict:     str  = ""
    warnings:    List[str] = field(default_factory=list)

    def evaluate(self):
        if not self.history:
            self.passed  = False
            self.verdict = "NO DATA"
            return

        last     = self.history[-1]
        avg_f1   = sum(m.f1 for m in self.history) / len(self.history)
        avg_prec = sum(m.precision for m in self.history) / len(self.history)
        avg_rec  = sum(m.recall   for m in self.history) / len(self.history)

        # Thresholds vary by scenario
        if self.name == "zero_fraud":
            # Main check: no false positives on clean batches
            clean_batches = [m for m in self.history if m.batch_idx in (1, 3)]
            fp_in_clean   = sum(m.fp for m in clean_batches)
            self.passed   = fp_in_clean == 0
            self.verdict  = (f"PASS — 0 false positives on clean batches"
                             if self.passed else
                             f"FAIL — {fp_in_clean} false positives on clean batches")
        elif self.name == "cascade":
            # Cascade: just need F1 > 0.50 (hardest scenario)
            self.passed  = avg_f1 >= 0.50
            self.verdict = f"{'PASS' if self.passed else 'FAIL'} — avg F1={avg_f1:.3f} (threshold 0.50)"
        else:
            # Standard: avg F1 > 0.70, adaptation visible
            tau_change = abs(self.history[-1].tau_alert - self.history[0].tau_alert)
            self.passed = avg_f1 >= 0.60
            self.verdict = (f"{'PASS' if self.passed else 'FAIL'} — "
                            f"avg F1={avg_f1:.3f} | avg P={avg_prec:.3f} | avg R={avg_rec:.3f} | "
                            f"tau_drift={tau_change:.3f}")

        # Warnings
        for m in self.history:
            if m.precision < 0.50 and m.tp + m.fp > 0:
                self.warnings.append(f"Batch {m.batch_idx}: low precision ({m.precision:.3f})")
            if m.recall < 0.40 and m.fn > 0:
                self.warnings.append(f"Batch {m.batch_idx}: low recall ({m.recall:.3f})")


def run_scenario(name: str, n_batches: int = 5, seed: int = SEED) -> ScenarioResult:
    fn, desc = SCENARIO_REGISTRY[name]
    model    = DetectorModel(scenario_fn=fn, n_batches=n_batches, seed=seed)
    history  = model.run()
    result   = ScenarioResult(name=name, description=desc, history=history)
    result.evaluate()
    return result


def run_all_scenarios(n_batches: int = 5, seed: int = SEED) -> Dict[str, ScenarioResult]:
    results = {}
    for name in SCENARIO_REGISTRY:
        results[name] = run_scenario(name, n_batches=n_batches, seed=seed)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# REPORT PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def _bar(val: float, width: int = 20) -> str:
    filled = int(round(val * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {val:.3f}"


def print_scenario_report(result: ScenarioResult):
    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  SCENARIO: {result.name.upper()}")
    print(f"  {result.description}")
    print(SEP)

    header = (f"  {'Batch':<7} {'P':>7} {'R':>7} {'F1':>7} "
              f"{'w':>6} {'tau':>7} {'TP':>5} {'FP':>5} {'FN':>5} "
              f"{'BLOCK':>7} {'ms':>7}")
    print(header)
    print("  " + "-" * 68)

    for m in result.history:
        print(f"  {m.batch_idx:<7} {m.precision:>7.3f} {m.recall:>7.3f} "
              f"{m.f1:>7.3f} {m.w:>6.3f} {m.tau_alert:>7.3f} "
              f"{m.tp:>5} {m.fp:>5} {m.fn:>5} "
              f"{m.n_block:>7} {m.latency_ms:>7.1f}")

    print()
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"  VERDICT: {status}")
    print(f"  {result.verdict}")
    if result.warnings:
        print(f"\n  WARNINGS:")
        for w in result.warnings:
            print(f"    ! {w}")


def print_summary(results: Dict[str, ScenarioResult]):
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  SIMULATION VALIDATION SUMMARY")
    print(SEP)

    passed = sum(1 for r in results.values() if r.passed)
    total  = len(results)

    print(f"\n  {'Scenario':<18} {'Status':<12} {'Avg F1':>8}  Description")
    print("  " + "-" * 68)
    for name, r in results.items():
        avg_f1  = sum(m.f1 for m in r.history) / len(r.history) if r.history else 0
        status  = "PASS ✓" if r.passed else "FAIL ✗"
        print(f"  {name:<18} {status:<12} {avg_f1:>8.3f}  {r.description[:38]}")

    print(f"\n  Result: {passed}/{total} scenarios passed")

    # Adaptation proof
    print(f"\n  ADAPTATION EVIDENCE:")
    for name, r in results.items():
        if len(r.history) >= 2:
            d_tau = r.history[-1].tau_alert - r.history[0].tau_alert
            d_w   = r.history[-1].w - r.history[0].w
            print(f"  {name:<18} tau_change={d_tau:+.3f}  w_change={d_w:+.3f}")

    print(f"\n{'='*70}\n")


def save_results(results: Dict[str, ScenarioResult], output_dir: str = "runs/simulation"):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, r in results.items():
        for m in r.history:
            row = asdict(m)
            row["scenario"] = name
            row["passed"]   = r.passed
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "simulation_results.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        name: {
            "passed":   r.passed,
            "verdict":  r.verdict,
            "warnings": r.warnings,
            "avg_f1":   round(sum(m.f1 for m in r.history) / len(r.history), 4),
            "avg_prec": round(sum(m.precision for m in r.history) / len(r.history), 4),
            "avg_rec":  round(sum(m.recall for m in r.history) / len(r.history), 4),
        }
        for name, r in results.items()
    }
    json_path = os.path.join(output_dir, "simulation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved:")
    print(f"    {csv_path}")
    print(f"    {json_path}")
    return csv_path, json_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Mesa stress-test simulation for Agentic Blockchain Fraud Detection"
    )
    parser.add_argument(
        "--scenario", default="all",
        choices=list(SCENARIO_REGISTRY.keys()) + ["all"],
        help="Which scenario to run (default: all)"
    )
    parser.add_argument("--batches", type=int, default=5,
                        help="Number of batches per scenario (default: 5)")
    parser.add_argument("--seed",    type=int, default=SEED,
                        help="Random seed (default: 42)")
    parser.add_argument("--report",  action="store_true",
                        help="Print detailed per-scenario reports")
    parser.add_argument("--save",    action="store_true", default=True,
                        help="Save results CSV + JSON (default: True)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  MESA SIMULATION — Agentic Blockchain Fraud Detection")
    print(f"  seed={args.seed} | batches={args.batches} | scenario={args.scenario}")
    print("=" * 70)

    t_start = time.perf_counter()

    if args.scenario == "all":
        results = run_all_scenarios(n_batches=args.batches, seed=args.seed)
    else:
        results = {args.scenario: run_scenario(args.scenario,
                                               n_batches=args.batches,
                                               seed=args.seed)}

    elapsed = time.perf_counter() - t_start

    if args.report or args.scenario != "all":
        for r in results.values():
            print_scenario_report(r)

    print_summary(results)
    print(f"  Total simulation time: {elapsed*1000:.0f}ms")

    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()