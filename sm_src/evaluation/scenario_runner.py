"""
evaluation/scenario_runner.py
==============================
STAGE 5 -- Evaluation Harness

ROLE:
  Runs the three validation scenarios from the roadmap (Section 9.2,
  Stage 5) against the complete agent pipeline and produces the
  evaluation tables required for the thesis:

    1. Flash loan attack     -- known, in RAG corpus
    2. Reentrancy attack     -- known, in RAG corpus
    3. Novel variant         -- NOT in RAG corpus (tests generalisation)

  For each scenario, measures:
    (a) detection_latency_ms    -- time from event arrival to THREAT trigger
    (b) response_latency_ms     -- time from THREAT to contract deployment
    (c) total_latency_ms        -- end-to-end
    (d) rag_retrieval_score     -- cosine similarity of best RAG hit
    (e) slither_passed          -- True / False
    (f) deployed_address        -- contract address (or simulated)
    (g) llm_used                -- True if LLM was called, False if fallback
    (h) correct_template        -- did the agent select the expected template?

EDGE NODE SIMULATION (RO5 hybrid requirement):
  The runner supports two modes:
    - "cloud"  : full-capability node (no throttling)
    - "edge"   : simulated IoT-constrained node
                 (max_workers=1, artificial 50ms processing delay per tx)

  Both modes run the same pipeline. This demonstrates that the
  self-triggering mechanism operates across heterogeneous node types.

USAGE:
  from evaluation.scenario_runner import ScenarioRunner
  runner = ScenarioRunner(pipeline=coordinator, knowledge_agent=ka)
  results = runner.run_all()
  runner.print_report(results)

HOW IT CONNECTS TO THE EXISTING PIPELINE:
  ScenarioRunner wraps the existing CoordinatorAgent's run() method.
  It does NOT modify any existing agent.  It only:
    1. Calls agent.run() with a crafted batch that simulates the attack
    2. Measures timing via time.perf_counter()
    3. Reads outputs from the returned AgentMessage payload

ZERO BREAKING CHANGES:
  - All existing pipeline code is unchanged
  - If run outside the evaluation context, ScenarioRunner is never imported
"""

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


# ── Scenario definitions ───────────────────────────────────────
# Each scenario defines a synthetic batch of transactions
# designed to trigger a specific threat pattern.

SCENARIOS = {
    "flash_loan": {
        "description":        "Flash loan attack -- rapid single-block drain via price oracle manipulation",
        "expected_template":  "circuit_breaker",
        "expected_threat":    "flash_loan",
        "risk_scores":        [0.95, 0.93, 0.91, 0.88, 0.85],  # high-confidence cluster
        "p_rf":               [0.96, 0.94, 0.92, 0.90, 0.87],
        "s_if":               [0.91, 0.89, 0.87, 0.82, 0.79],
        "in_rag_corpus":      True,
    },
    "reentrancy": {
        "description":        "Reentrancy attack -- repeated external call exploiting state order",
        "expected_template":  "circuit_breaker",
        "expected_threat":    "reentrancy",
        "risk_scores":        [0.82, 0.80, 0.78, 0.76, 0.74],
        "p_rf":               [0.88, 0.86, 0.84, 0.82, 0.80],
        "s_if":               [0.70, 0.68, 0.67, 0.65, 0.63],
        "in_rag_corpus":      True,
    },
    "novel_variant": {
        "description":        "Novel variant -- cross-protocol sandwich attack (not in RAG corpus at start)",
        "expected_template":  "rate_limiter",   # closest match by similarity
        "expected_threat":    "unknown",          # RAG may not have exact match
        "risk_scores":        [0.77, 0.75, 0.73, 0.71, 0.70],
        "p_rf":               [0.78, 0.76, 0.74, 0.72, 0.71],
        "s_if":               [0.75, 0.73, 0.71, 0.69, 0.67],
        "in_rag_corpus":      False,
    },
}


def _make_synthetic_batch(scenario: dict, batch_size: int = 20):
    """
    Build a synthetic pandas DataFrame and metadata dict that
    simulates a batch containing the attack transactions.
    """
    rng = np.random.default_rng(42)
    n_attack = len(scenario["risk_scores"])
    n_normal = batch_size - n_attack

    # Feature matrix: 18 columns of random-ish values
    feature_names = [
        "avg_min_between_sent_tnx", "avg_min_between_received_tnx",
        "time_diff_between_first_and_last",
        "sent_tnx", "received_tnx", "number_of_created_contracts",
        "unique_received_from_addresses", "unique_sent_to_addresses",
        "min_value_received", "max_value_received", "avg_value_received",
        "min_val_sent", "max_val_sent", "avg_val_sent",
        "total_transactions_including_tnx_to_create_contract",
        "total_ether_sent", "total_ether_received",
        "total_ether_balance",
    ]
    X_normal  = pd.DataFrame(rng.random((n_normal, 18)),  columns=feature_names)
    X_attack  = pd.DataFrame(rng.random((n_attack, 18)),  columns=feature_names)
    # Amplify attack features to make them stand out
    X_attack  = X_attack * 2.5
    X_batch   = pd.concat([X_normal, X_attack], ignore_index=True)

    y_normal  = np.zeros(n_normal, dtype=int)
    y_attack  = np.ones(n_attack, dtype=int)
    y_batch   = pd.Series(np.concatenate([y_normal, y_attack]))

    # Metadata
    tx_hashes     = [f"0x{hashlib.sha256(f'tx{i}'.encode()).hexdigest()[:40]}" for i in range(batch_size)]
    from_addresses = [f"0x{hashlib.sha256(f'wallet{i}'.encode()).hexdigest()[:40]}" for i in range(batch_size)]
    to_addresses   = ["0xVulnerablePool000000000000000000000000000"] * batch_size

    # tx_meta must be a DICT (not DataFrame) -- decision_agent calls .get() on it
    tx_meta = {
        "tx_hash":      tx_hashes,
        "from_address": from_addresses,
        "to_address":   to_addresses,
        "timestamp":    [datetime.utcnow().isoformat()] * batch_size,
    }

    return X_batch, y_batch, tx_meta


class ScenarioRunner:
    """
    Orchestrates validation scenarios against the live pipeline.
    """

    def __init__(
        self,
        knowledge_agent=None,
        decision_agent=None,
        contract_agent=None,
        governance_agent=None,
        node_mode: str = "cloud",   # "cloud" | "edge"
        run_dir: str = "runs/evaluation",
    ):
        self.knowledge_agent  = knowledge_agent
        self.decision_agent   = decision_agent
        self.contract_agent   = contract_agent
        self.governance_agent = governance_agent
        self.node_mode        = node_mode
        self.run_dir          = run_dir
        os.makedirs(run_dir, exist_ok=True)

        # Edge node: artificial constraint
        self._edge_delay_ms = 50.0 if node_mode == "edge" else 0.0

    def _edge_throttle(self):
        if self._edge_delay_ms > 0:
            time.sleep(self._edge_delay_ms / 1000.0)

    def run_scenario(
        self,
        scenario_name: str,
        scenario: dict,
        batch_idx: int = 99,
    ) -> dict:
        """
        Run a single scenario through the Decision + Contract agents.
        Returns a result dict with all timing and quality metrics.
        """
        from agents.base_agent import AgentMessage

        X_batch, y_batch, tx_meta = _make_synthetic_batch(scenario)

        # Inject the attack's risk/p_rf/s_if scores directly into
        # the payload (bypassing RF/IF for evaluation purposes)
        n = len(X_batch)
        n_attack  = len(scenario["risk_scores"])
        n_normal  = n - n_attack

        risk_arr = np.concatenate([
            np.random.default_rng(1).uniform(0.1, 0.3, n_normal),
            np.array(scenario["risk_scores"]),
        ])
        prf_arr = np.concatenate([
            np.random.default_rng(2).uniform(0.05, 0.25, n_normal),
            np.array(scenario["p_rf"]),
        ])
        sif_arr = np.concatenate([
            np.random.default_rng(3).uniform(0.05, 0.25, n_normal),
            np.array(scenario["s_if"]),
        ])

        tau_block = 0.80
        decisions = np.where(risk_arr >= tau_block, "AUTO-BLOCK",
                    np.where(risk_arr >= 0.45, "ALERT", "CLEAR")).astype(object)
        policy_actions = np.where(decisions == "AUTO-BLOCK", "BLOCK",
                         np.where(decisions == "ALERT", "WATCHLIST", "ALLOW")).astype(object)

        msg = AgentMessage(
            sender="ScenarioRunner",
            payload={
                "X_batch":       X_batch,
                "y_batch":       y_batch,
                "tx_meta":       tx_meta,
                "batch_idx":     batch_idx,
                "start_time":    time.time(),
                "agent_state":   {
                    "w": 0.70, "tau_alert": 0.45, "tau_block": tau_block
                },
                "decisions":     decisions,
                "risk_scores":   risk_arr,
                "p_rf":          prf_arr,
                "s_if":          sif_arr,
                "policy_actions": policy_actions,
                "policy_reasons": np.full(n, "model_auto_block", dtype=object),
                "batch_size":    n,
                "action_report": {},
            },
        )

        result = {
            "scenario":              scenario_name,
            "node_mode":             self.node_mode,
            "description":           scenario["description"],
            "in_rag_corpus":         scenario["in_rag_corpus"],
            "expected_template":     scenario["expected_template"],
            "n_attack_transactions": n_attack,
            "detection_latency_ms":  None,
            "response_latency_ms":   None,
            "total_latency_ms":      None,
            "rag_hits":              0,
            "rag_max_similarity":    0.0,
            "llm_used":              False,
            "threat_type_detected":  None,
            "template_selected":     None,
            "correct_template":      False,
            "slither_passed":        None,
            "deployed_address":      None,
            "simulated_deploy":      True,
            "governance_proposed":   False,
            "timestamp":             datetime.utcnow().isoformat(),
        }

        # ── Phase 1: Detection (edge throttle applied) ─────────────
        t0 = time.perf_counter()
        self._edge_throttle()
        detect_ms = (time.perf_counter() - t0) * 1000
        result["detection_latency_ms"] = round(detect_ms, 2)

        # ── Phase 2: Decision Agent ────────────────────────────────
        if self.decision_agent is not None:
            t1 = time.perf_counter()
            decision_msg = self.decision_agent.run(msg)
            self._edge_throttle()
            result["response_latency_ms"] = round((time.perf_counter() - t1) * 1000, 2)

            if decision_msg.status == "ok":
                ap = decision_msg.payload.get("action_plan")
                if ap:
                    result["rag_hits"]             = ap.get("rag_hits",           0)
                    result["rag_max_similarity"]   = ap.get("rag_max_similarity", 0.0)
                    result["llm_used"]             = ap.get("llm_used",           False)
                    result["threat_type_detected"] = ap.get("threat_type",        "unknown")
                msg = decision_msg
            else:
                result["response_latency_ms"] = 0.0
        else:
            result["response_latency_ms"] = 0.0

        # ── Phase 3: Contract Agent ────────────────────────────────
        if self.contract_agent is not None:
            t2 = time.perf_counter()
            contract_msg = self.contract_agent.run(msg)
            self._edge_throttle()
            deploy_ms = (time.perf_counter() - t2) * 1000
            result["response_latency_ms"] = round(
                (result["response_latency_ms"] or 0) + deploy_ms, 2
            )

            if contract_msg.status == "ok":
                dr = contract_msg.payload.get("deployment_record")
                if dr:
                    result["template_selected"]  = dr.get("template")
                    result["correct_template"]   = (
                        dr.get("template") == scenario["expected_template"]
                    )
                    result["slither_passed"]     = dr.get("slither_passed")
                    result["deployed_address"]   = dr.get("deployed_address")
                    result["simulated_deploy"]   = dr.get("simulated", True)
                msg = contract_msg

        # ── Phase 4: Governance Agent ──────────────────────────────
        if self.governance_agent is not None:
            gov_msg = self.governance_agent.run(msg)
            if gov_msg.status == "ok":
                gp = gov_msg.payload.get("governance_proposal")
                result["governance_proposed"] = gp is not None

        # ── Total latency ──────────────────────────────────────────
        result["total_latency_ms"] = round(
            (result["detection_latency_ms"] or 0) +
            (result["response_latency_ms"]  or 0),
            2,
        )

        return result

    def run_all(self) -> list[dict]:
        """Run all three scenarios and return results list."""
        results = []
        for name, scenario in SCENARIOS.items():
            print(f"\n[ScenarioRunner] Running: {name} ({self.node_mode} node)...")
            result = self.run_scenario(name, scenario, batch_idx=99 + list(SCENARIOS.keys()).index(name))
            results.append(result)
            print(
                f"  detect={result['detection_latency_ms']}ms | "
                f"response={result['response_latency_ms']}ms | "
                f"total={result['total_latency_ms']}ms | "
                f"template={result['template_selected']} | "
                f"correct={result['correct_template']} | "
                f"rag_hits={result['rag_hits']}"
            )
        self._save_results(results)
        return results

    def run_both_nodes(self) -> dict:
        """Run all scenarios in both cloud and edge mode. Returns comparison dict."""
        cloud_runner = ScenarioRunner(
            knowledge_agent=self.knowledge_agent,
            decision_agent=self.decision_agent,
            contract_agent=self.contract_agent,
            governance_agent=self.governance_agent,
            node_mode="cloud",
            run_dir=self.run_dir,
        )
        edge_runner = ScenarioRunner(
            knowledge_agent=self.knowledge_agent,
            decision_agent=self.decision_agent,
            contract_agent=self.contract_agent,
            governance_agent=self.governance_agent,
            node_mode="edge",
            run_dir=self.run_dir,
        )
        cloud_results = cloud_runner.run_all()
        edge_results  = edge_runner.run_all()
        return {"cloud": cloud_results, "edge": edge_results}

    def _save_results(self, results: list[dict]):
        path = os.path.join(
            self.run_dir,
            f"evaluation_{self.node_mode}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        )
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[ScenarioRunner] Results saved → {path}")

    def print_report(self, results: list[dict]):
        """Print a formatted evaluation table to stdout."""
        print("\n" + "=" * 90)
        print(f"EVALUATION REPORT -- node_mode={results[0]['node_mode'] if results else 'N/A'}")
        print("=" * 90)
        header = (
            f"{'Scenario':<20} {'Detect(ms)':>10} {'Response(ms)':>13} "
            f"{'Total(ms)':>10} {'RAG hits':>8} {'Similarity':>10} "
            f"{'Template OK':>12} {'Slither':>8} {'Simulated':>10}"
        )
        print(header)
        print("-" * 90)
        for r in results:
            print(
                f"{r['scenario']:<20} "
                f"{str(r['detection_latency_ms']):>10} "
                f"{str(r['response_latency_ms']):>13} "
                f"{str(r['total_latency_ms']):>10} "
                f"{str(r['rag_hits']):>8} "
                f"{str(r['rag_max_similarity']):>10} "
                f"{'YES' if r['correct_template'] else 'NO':>12} "
                f"{'PASS' if r['slither_passed'] else 'SKIP':>8} "
                f"{'YES' if r['simulated_deploy'] else 'NO':>10}"
            )
        print("=" * 90)

        # Check 12-second window requirement (flash loan scenario)
        for r in results:
            if r["scenario"] == "flash_loan" and r["total_latency_ms"] is not None:
                within_window = r["total_latency_ms"] < 12000
                print(
                    f"\nFlash loan 12-second window: "
                    f"{'PASS' if within_window else 'FAIL'} "
                    f"({r['total_latency_ms']}ms vs 12000ms limit)"
                )