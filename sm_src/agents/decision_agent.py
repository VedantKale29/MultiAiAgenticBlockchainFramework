"""
agents/decision_agent.py
=========================
STAGE 3 -- DecisionAgent

ROLE IN FRAMEWORK (Section 4, Layer 3):
  Receives the trigger payload from FusionAgent / MonitoringAgent.
  Queries RAG for similar past incidents.
  Calls LLM with threat context + retrieved evidence.
  Outputs a structured ActionPlan dict.

MAPS TO:
  RO4 -- the agentic AI reasoning layer; LLM-based decision with
         RAG-grounded context is the core of 'agentic AI-driven
         blockchain for cybersecurity'

HOW IT FITS IN THE EXISTING PIPELINE:
  DecisionAgent is called AFTER MonitoringAgent and BEFORE
  AdaptationAgent, but ONLY for batches that contain at least
  one AUTO-BLOCK decision.  For CLEAR-only batches it is skipped
  (zero overhead on the normal path).

  Pipeline slot (inside CoordinatorAgent's batch loop):
    ... monitoring_agent.run(current_msg)  ← existing
    ... decision_agent.run(current_msg)    ← NEW (Stage 3)
    ... adaptation_agent.run(...)          ← existing

ZERO BREAKING CHANGES:
  - If the Anthropic SDK is not installed or the API key is absent,
    the agent falls back to a rule-based ActionPlan and logs a
    warning.  The existing pipeline runs unchanged.
  - FraudKnowledgeAgent is optional; if absent, the LLM prompt
    is sent without RAG context.
  - All errors are caught inside _run(); a failed call returns
    a default ActionPlan, never an error status that would stop
    the pipeline.

ActionPlan schema (dict):
  {
    "threat_type":          str,   # e.g. "flash_loan", "reentrancy", "phishing"
    "severity":             str,   # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    "recommended_template": str,   # template key from contract library
    "parameters": {
        "target_address":   str,   # address to protect / block / rate-limit
        "threshold":        float, # numeric threshold for the contract
        "attacker_address": str,   # address of the threat actor (if known)
    },
    "reasoning":            str,   # LLM's explanation (for audit log)
    "rag_hits":             int,   # how many similar past events retrieved
    "rag_max_similarity":   float, # cosine similarity of best match
    "llm_used":             bool,  # True if LLM was called, False for fallback
  }

INSTALL:
  pip install anthropic        # LLM calls
  pip install chromadb sentence-transformers   # RAG (Stage 1 dependency)
"""

import os
import json
import re
from typing import Optional

from agents.base_agent import BaseAgent, AgentMessage
import numpy as np


# ── Threat-type heuristics used in rule-based fallback ────────
_THREAT_HEURISTICS = {
    "flash_loan":  lambda s, r: s >= 0.85 and r >= 0.90,
    "reentrancy":  lambda s, r: 0.70 <= s < 0.85 and r >= 0.85,
    "phishing":    lambda s, r: s >= 0.70 and r < 0.85,
}

# ── Template key → description mapping ────────────────────────
_TEMPLATE_DESCRIPTIONS = {
    "circuit_breaker":  "Pauses the target contract, halting all state-changing calls.",
    "address_blocklist":"Blocks a specific wallet address from interacting with the protocol.",
    "rate_limiter":     "Throttles per-block transaction volume to prevent flash-loan draining.",
}

# ── Severity thresholds ────────────────────────────────────────
def _score_to_severity(risk_score: float) -> str:
    if risk_score >= 0.90: return "CRITICAL"
    if risk_score >= 0.75: return "HIGH"
    if risk_score >= 0.60: return "MEDIUM"
    return "LOW"


class DecisionAgent(BaseAgent):
    """
    LLM-powered decision agent with RAG-grounded threat reasoning.
    Gracefully degrades to rule-based decisions if LLM unavailable.
    """

    def __init__(
        self,
        knowledge_agent=None,    # FraudKnowledgeAgent instance (Stage 1)
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
    ):
        super().__init__(name="DecisionAgent")
        self.knowledge_agent   = knowledge_agent
        self.model             = model
        self.max_tokens        = max_tokens
        self._anthropic_client = None

        # Try to initialise Anthropic client
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
                self.logger.info(
                    f"[{self.name}] Anthropic client ready (model={self.model})"
                )
            except ImportError:
                self.logger.warning(
                    f"[{self.name}] anthropic SDK not installed -- "
                    f"falling back to rule-based decisions. "
                    f"Install with: pip install anthropic"
                )
        else:
            self.logger.warning(
                f"[{self.name}] ANTHROPIC_API_KEY not set -- "
                f"falling back to rule-based decisions."
            )

    # ═══════════════════════════════════════════════════════════
    # RULE-BASED FALLBACK
    # ═══════════════════════════════════════════════════════════
    def _rule_based_plan(
        self,
        mean_risk: float,
        mean_p_rf: float,
        top_wallet: str,
        to_address: str,
        n_blocked: int,
    ) -> dict:
        """
        Fast heuristic ActionPlan when LLM is unavailable.
        Covers the three most common threat patterns.
        """
        threat_type = "phishing"
        for name, test in _THREAT_HEURISTICS.items():
            if test(mean_risk, mean_p_rf):
                threat_type = name
                break

        template_map = {
            "flash_loan":  "circuit_breaker",
            "reentrancy":  "circuit_breaker",
            "phishing":    "address_blocklist",
        }

        return {
            "threat_type":          threat_type,
            "severity":             _score_to_severity(mean_risk),
            "recommended_template": template_map[threat_type],
            "parameters": {
                "target_address":   to_address,
                "threshold":        round(float(mean_risk), 4),
                "attacker_address": top_wallet,
            },
            "reasoning": (
                f"Rule-based decision (LLM unavailable). "
                f"Mean risk={mean_risk:.3f}, RF={mean_p_rf:.3f}. "
                f"Pattern matched: {threat_type}. "
                f"{n_blocked} transactions blocked this batch."
            ),
            "rag_hits":           0,
            "rag_max_similarity": 0.0,
            "llm_used":           False,
        }

    # ═══════════════════════════════════════════════════════════
    # RAG CONTEXT RETRIEVAL
    # ═══════════════════════════════════════════════════════════
    def _get_rag_context(
        self,
        risk_score: float,
        p_rf: float,
        s_if: float,
        decision: str,
        wallet: str,
        n_results: int = 3,
    ) -> tuple[str, int, float]:
        """
        Returns (rag_context_str, n_hits, max_similarity).
        Returns ("", 0, 0.0) if RAG unavailable.
        """
        if self.knowledge_agent is None:
            return "", 0, 0.0
        try:
            similar = self.knowledge_agent.query_similar(
                query_text=(
                    f"risk_score={risk_score:.3f} p_rf={p_rf:.3f} "
                    f"s_if={s_if:.3f} decision={decision} wallet={wallet}"
                ),
                n_results=n_results,
                confirmed_only=True,
            )
            if not similar:
                return "", 0, 0.0
            lines = []
            for i, item in enumerate(similar, 1):
                m = item["metadata"]
                sim = round(1 - item["distance"], 3)
                lines.append(
                    f"[{i}] similarity={sim} | "
                    f"batch={m.get('batch','?')} | "
                    f"decision={m.get('decision','?')} | "
                    f"action={m.get('policy_action','?')} | "
                    f"risk={float(m.get('risk_score',0)):.3f} | "
                    f"label={'fraud' if m.get('y_true')=='1' else 'fp'}"
                )
            max_sim = round(1 - similar[0]["distance"], 3)
            return "\n".join(lines), len(similar), max_sim
        except Exception as e:
            self.logger.warning(f"[{self.name}] RAG query failed: {e}")
            return "", 0, 0.0

    # ═══════════════════════════════════════════════════════════
    # LLM CALL
    # ═══════════════════════════════════════════════════════════
    _SYSTEM_PROMPT = """You are a blockchain fraud detection decision engine.
You receive threat detection data and similar past incidents retrieved from a knowledge base.
You must output ONLY valid JSON -- no markdown, no preamble, no explanation outside the JSON.

Output exactly this structure:
{
  "threat_type": "<flash_loan|reentrancy|phishing|wash_trading|unknown>",
  "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "recommended_template": "<circuit_breaker|address_blocklist|rate_limiter>",
  "parameters": {
    "target_address": "<address of the contract or wallet to protect>",
    "threshold": <float between 0 and 1>,
    "attacker_address": "<address of the threat actor>"
  },
  "reasoning": "<one sentence explaining the decision>"
}

Template selection guide:
- circuit_breaker: use for flash loans, reentrancy -- halts the target contract
- address_blocklist: use for phishing, wash trading -- blocks a specific wallet
- rate_limiter: use when volume throttling is the correct response"""

    def _call_llm(self, user_prompt: str) -> Optional[dict]:
        """
        Call Anthropic API, parse JSON response.
        Returns None on any failure.
        """
        if self._anthropic_client is None:
            return None
        try:
            response = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            self.logger.warning(f"[{self.name}] LLM call failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # BUILD USER PROMPT
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _build_prompt(
        batch_idx: int,
        n_blocked: int,
        mean_risk: float,
        mean_p_rf: float,
        mean_s_if: float,
        top_wallet: str,
        to_address: str,
        rag_context: str,
    ) -> str:
        rag_section = (
            f"\nSimilar past incidents from knowledge base:\n{rag_context}"
            if rag_context else
            "\nNo similar past incidents found in knowledge base."
        )
        return (
            f"Batch {batch_idx + 1} threat detection summary:\n"
            f"  Transactions blocked: {n_blocked}\n"
            f"  Mean risk score: {mean_risk:.4f}\n"
            f"  Mean RF probability: {mean_p_rf:.4f}\n"
            f"  Mean IF anomaly score: {mean_s_if:.4f}\n"
            f"  Primary threat wallet: {top_wallet}\n"
            f"  Target contract address: {to_address}\n"
            f"{rag_section}\n\n"
            f"Determine: threat_type, severity, recommended_template, parameters, reasoning."
        )

    # ═══════════════════════════════════════════════════════════
    # MAIN _run
    # ═══════════════════════════════════════════════════════════
    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Only activates when the batch contains AUTO-BLOCK decisions.
        For all-CLEAR batches: passes through with action_plan=None.
        """
        decisions    = np.asarray(msg.payload.get("decisions",    []), dtype=object)
        risk_scores  = np.asarray(msg.payload.get("risk_scores",  []), dtype=float)
        p_rf         = np.asarray(msg.payload.get("p_rf",         []), dtype=float)
        s_if         = np.asarray(msg.payload.get("s_if",         []), dtype=float)
        batch_idx    = msg.payload.get("batch_idx", 0)
        tx_meta      = msg.payload.get("tx_meta",   {})
        policy_actions = np.asarray(
            msg.payload.get("policy_actions", []), dtype=object
        )

        # Only reason about batches with BLOCK decisions
        block_mask = np.asarray(policy_actions) == "BLOCK" if len(policy_actions) > 0 else np.array([], dtype=bool)
        n_blocked  = int(np.sum(block_mask))

        if n_blocked == 0:
            self.logger.info(
                f"[{self.name}] Batch {batch_idx+1} -- no BLOCK decisions, skipping."
            )
            return AgentMessage(
                sender=self.name,
                payload={**msg.payload, "action_plan": None},
                status="ok",
            )

        # Compute summary statistics over blocked transactions
        blocked_risks  = risk_scores[block_mask] if len(risk_scores) > 0 and np.any(block_mask) else risk_scores
        blocked_prf    = p_rf[block_mask]         if len(p_rf)        > 0 and np.any(block_mask) else p_rf
        blocked_sif    = s_if[block_mask]         if len(s_if)        > 0 and np.any(block_mask) else s_if

        mean_risk = float(np.mean(blocked_risks)) if len(blocked_risks) > 0 else 0.5
        mean_p_rf = float(np.mean(blocked_prf))   if len(blocked_prf)  > 0 else 0.5
        mean_s_if = float(np.mean(blocked_sif))   if len(blocked_sif)  > 0 else 0.5

        # Highest-risk wallet address
        from_addrs = tx_meta.get("from_address", []) if tx_meta else []
        to_addrs   = tx_meta.get("to_address",   []) if tx_meta else []
        if len(risk_scores) > 0 and len(from_addrs) > 0:
            top_idx    = int(np.argmax(risk_scores))
            top_wallet = str(from_addrs[top_idx]) if top_idx < len(from_addrs) else "unknown"
            to_addr    = str(to_addrs[top_idx])   if top_idx < len(to_addrs)   else "unknown"
        else:
            top_wallet = "unknown"
            to_addr    = "unknown"

        # RAG context retrieval
        rag_ctx, rag_hits, rag_max_sim = self._get_rag_context(
            risk_score=mean_risk, p_rf=mean_p_rf, s_if=mean_s_if,
            decision="AUTO-BLOCK", wallet=top_wallet,
        )

        action_plan: dict

        if self._anthropic_client is not None:
            prompt  = self._build_prompt(
                batch_idx, n_blocked, mean_risk, mean_p_rf, mean_s_if,
                top_wallet, to_addr, rag_ctx,
            )
            llm_result = self._call_llm(prompt)

            if llm_result and isinstance(llm_result, dict):
                # Validate required keys; fill missing with defaults
                action_plan = {
                    "threat_type":          llm_result.get("threat_type", "unknown"),
                    "severity":             llm_result.get("severity", _score_to_severity(mean_risk)),
                    "recommended_template": llm_result.get("recommended_template", "address_blocklist"),
                    "parameters": {
                        "target_address":   llm_result.get("parameters", {}).get("target_address", to_addr),
                        "threshold":        float(llm_result.get("parameters", {}).get("threshold", mean_risk)),
                        "attacker_address": llm_result.get("parameters", {}).get("attacker_address", top_wallet),
                    },
                    "reasoning":            llm_result.get("reasoning", "LLM decision"),
                    "rag_hits":             rag_hits,
                    "rag_max_similarity":   rag_max_sim,
                    "llm_used":             True,
                }
            else:
                # LLM returned invalid JSON -- use rule-based fallback
                action_plan = self._rule_based_plan(
                    mean_risk, mean_p_rf, top_wallet, to_addr, n_blocked
                )
                action_plan["rag_hits"]           = rag_hits
                action_plan["rag_max_similarity"]  = rag_max_sim
        else:
            # No LLM client -- rule-based fallback
            action_plan = self._rule_based_plan(
                mean_risk, mean_p_rf, top_wallet, to_addr, n_blocked
            )
            action_plan["rag_hits"]           = rag_hits
            action_plan["rag_max_similarity"]  = rag_max_sim

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"threat={action_plan['threat_type']} "
            f"severity={action_plan['severity']} "
            f"template={action_plan['recommended_template']} "
            f"n_blocked={n_blocked} "
            f"rag_hits={action_plan['rag_hits']} "
            f"llm={action_plan['llm_used']}"
        )

        return AgentMessage(
            sender=self.name,
            payload={**msg.payload, "action_plan": action_plan},
            status="ok",
        )