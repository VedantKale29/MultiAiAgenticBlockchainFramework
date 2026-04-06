"""
agents/decision_agent.py
=========================
STAGE 3 -- DecisionAgent  (UPGRADED: tri-backend LLM support)

ROLE IN FRAMEWORK (Section 4, Layer 3):
  Receives the trigger payload from FusionAgent / MonitoringAgent.
  Queries RAG for similar past incidents.
  Calls LLM with threat context + retrieved evidence.
  Outputs a structured ActionPlan dict.

MAPS TO:
  RO4 -- the agentic AI reasoning layer; LLM-based decision with
         RAG-grounded context is the core of 'agentic AI-driven
         blockchain for cybersecurity'

═══════════════════════════════════════════════════════════════
LLM BACKEND SELECTION (auto-detected, priority order)
═══════════════════════════════════════════════════════════════

  Priority 1 — AWS Bedrock  (IS_SAGEMAKER=True OR USE_BEDROCK=true)
    - Uses boto3 (already installed in SageMaker)
    - No separate API key — bills to your AWS account
    - Set env: USE_BEDROCK=true  BEDROCK_REGION=us-east-1
    - Model: anthropic.claude-3-sonnet-20240229-v1:0  (cheapest ~$0.001/run)

  Priority 2 — Anthropic Direct API  (ANTHROPIC_API_KEY set)
    - Uses anthropic SDK
    - Set env: ANTHROPIC_API_KEY=sk-ant-...
    - Model: anthropic.claude-3-sonnet-20240229-v1:0  (configurable)

  Priority 3 — Ollama local  (OLLAMA_URL reachable)
    - Free, no API key, runs on your laptop
    - Start: ollama pull llama3.2 && ollama serve
    - Set env: OLLAMA_URL=http://localhost:11434  OLLAMA_MODEL=llama3.2

  Priority 4 — Rule-based fallback  (always works, no LLM)
    - Zero cost, deterministic, fast
    - Used when none of the above are available

HOW TO CONFIGURE FOR EACH ENVIRONMENT:

  Local development (Windows/Mac laptop):
    set OLLAMA_URL=http://localhost:11434
    set OLLAMA_MODEL=llama3.2
    python run_pipeline.py

  AWS SageMaker training job:
    # No env vars needed -- Bedrock auto-detected via IS_SAGEMAKER
    # OR explicitly:
    set USE_BEDROCK=true
    set BEDROCK_REGION=us-east-1
    set BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

  Direct Anthropic API (any environment):
    set ANTHROPIC_API_KEY=sk-ant-...
    python run_pipeline.py

═══════════════════════════════════════════════════════════════
ActionPlan schema (dict):
  {
    "threat_type":          str,   # flash_loan | reentrancy | phishing | etc.
    "severity":             str,   # LOW | MEDIUM | HIGH | CRITICAL
    "recommended_template": str,   # circuit_breaker | address_blocklist | rate_limiter
    "parameters": {
        "target_address":   str,
        "threshold":        float,
        "attacker_address": str,
    },
    "reasoning":            str,   # LLM explanation (for audit log)
    "rag_hits":             int,
    "rag_max_similarity":   float,
    "llm_used":             bool,
    "llm_backend":          str,   # "bedrock" | "anthropic" | "ollama" | "rule_based"
  }
"""

import os
import json
import re
import requests
from typing import Optional

from agents.base_agent import BaseAgent, AgentMessage
import numpy as np
import config as config


# ── Threat-type heuristics used in rule-based fallback ─────────
_THREAT_HEURISTICS = {
    "flash_loan":  lambda s, r: s >= 0.85 and r >= 0.90,
    "reentrancy":  lambda s, r: 0.70 <= s < 0.85 and r >= 0.85,
    "phishing":    lambda s, r: s >= 0.70 and r < 0.85,
}

# ── Template key → description mapping ─────────────────────────
_TEMPLATE_DESCRIPTIONS = {
    "circuit_breaker":   "Pauses the target contract, halting all state-changing calls.",
    "address_blocklist": "Blocks a specific wallet address from interacting with the protocol.",
    "rate_limiter":      "Throttles per-block transaction volume to prevent flash-loan draining.",
}

# ── Severity thresholds ─────────────────────────────────────────
def _score_to_severity(risk_score: float) -> str:
    if risk_score >= 0.90: return "CRITICAL"
    if risk_score >= 0.75: return "HIGH"
    if risk_score >= 0.60: return "MEDIUM"
    return "LOW"


# ══════════════════════════════════════════════════════════════════
# BACKEND CONSTANTS  (overridable via environment variables)
# ══════════════════════════════════════════════════════════════════

# AWS Bedrock
_BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
_BEDROCK_MODEL  = os.getenv(
    "BEDROCK_MODEL",
    "anthropic.claude-3-sonnet-20240229-v1:0"   # cheapest, fastest on Bedrock
)

# Ollama
_OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "llama3.2")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))   # seconds

# Force flags
_USE_BEDROCK    = os.getenv("USE_BEDROCK", "").lower() in ("true", "1", "yes")
_USE_OLLAMA     = os.getenv("USE_OLLAMA",  "").lower() in ("true", "1", "yes")


class DecisionAgent(BaseAgent):
    """
    LLM-powered decision agent with RAG-grounded threat reasoning.

    Backend priority (auto-detected):
      1. AWS Bedrock  -- SageMaker environment or USE_BEDROCK=true
      2. Anthropic    -- ANTHROPIC_API_KEY set
      3. Ollama       -- OLLAMA_URL reachable locally
      4. Rule-based   -- always-available deterministic fallback
    """

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

    def __init__(
        self,
        knowledge_agent=None,
        anthropic_api_key: Optional[str] = None,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens: int = 512,
        # explicit overrides (optional -- env vars take priority)
        use_bedrock: bool = False,
        use_ollama:  bool = False,
    ):
        super().__init__(name="DecisionAgent")
        self.knowledge_agent = knowledge_agent
        self.model           = model
        self.max_tokens      = max_tokens

        # Backend state
        self._backend          = "rule_based"   # active backend name
        self._anthropic_client = None           # anthropic SDK client
        self._bedrock_client   = None           # boto3 bedrock-runtime client
        self._ollama_available = False          # True if Ollama responded to ping

        self._init_backend(anthropic_api_key, use_bedrock, use_ollama)

    # ═══════════════════════════════════════════════════════════
    # BACKEND INITIALISATION
    # ═══════════════════════════════════════════════════════════

    def _init_backend(
        self,
        anthropic_api_key: Optional[str],
        use_bedrock: bool,
        use_ollama: bool,
    ):
        """
        Detect and initialise the best available LLM backend.
        Priority: Bedrock > Anthropic > Ollama > rule-based
        """
        # ── 1. AWS Bedrock ──────────────────────────────────────
        want_bedrock = use_bedrock or _USE_BEDROCK or config.IS_SAGEMAKER
        if want_bedrock:
            if self._init_bedrock():
                return   # success -- stop here

        # ── 2. Anthropic direct API ─────────────────────────────
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and not _USE_OLLAMA and not use_ollama:
            if self._init_anthropic(api_key):
                return

        # ── 3. Ollama local ─────────────────────────────────────
        want_ollama = use_ollama or _USE_OLLAMA or bool(os.getenv("OLLAMA_URL"))
        if want_ollama or (not api_key and not want_bedrock):
            if self._init_ollama():
                return

        # ── 4. Rule-based fallback ──────────────────────────────
        self._backend = "rule_based"
        self.logger.warning(
            f"[{self.name}] No LLM backend available. Using rule-based fallback.\n"
            f"  To use Bedrock  : set USE_BEDROCK=true (or run in SageMaker)\n"
            f"  To use Anthropic: set ANTHROPIC_API_KEY=sk-ant-...\n"
            f"  To use Ollama   : ollama pull llama3.2 && ollama serve"
        )

    def _init_bedrock(self) -> bool:
        """Initialise AWS Bedrock client. Returns True on success."""
        try:
            import boto3
            # Connectivity test using a cheap list call
            boto3.client("bedrock", region_name=_BEDROCK_REGION).list_foundation_models(
                byProvider="Anthropic"
            )
            self._bedrock_client = boto3.client(
                "bedrock-runtime", region_name=_BEDROCK_REGION
            )
            self._backend = "bedrock"
            self.logger.info(
                f"[{self.name}] Backend: AWS Bedrock | "
                f"region={_BEDROCK_REGION} | model={_BEDROCK_MODEL}"
            )
            return True
        except ImportError:
            self.logger.warning(
                f"[{self.name}] boto3 not installed. Install: pip install boto3"
            )
        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Bedrock init failed: {e} "
                f"(check IAM permissions: bedrock:InvokeModel)"
            )
        return False

    def _init_anthropic(self, api_key: str) -> bool:
        """Initialise Anthropic SDK client. Returns True on success."""
        try:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            self._backend          = "anthropic"
            self.logger.info(
                f"[{self.name}] Backend: Anthropic API | model={self.model}"
            )
            return True
        except ImportError:
            self.logger.warning(
                f"[{self.name}] anthropic SDK not installed. "
                f"Install: pip install anthropic"
            )
        except Exception as e:
            self.logger.warning(f"[{self.name}] Anthropic init failed: {e}")
        return False

    def _init_ollama(self) -> bool:
        """Ping Ollama server. Returns True if reachable."""
        try:
            resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if not any(_OLLAMA_MODEL in m for m in models):
                    self.logger.warning(
                        f"[{self.name}] Ollama running but model '{_OLLAMA_MODEL}' "
                        f"not pulled yet. Run: ollama pull {_OLLAMA_MODEL}. "
                        f"Available models: {models}"
                    )
                self._ollama_available = True
                self._backend          = "ollama"
                self.logger.info(
                    f"[{self.name}] Backend: Ollama local | "
                    f"url={_OLLAMA_URL} | model={_OLLAMA_MODEL}"
                )
                return True
        except Exception as e:
            self.logger.info(
                f"[{self.name}] Ollama not reachable at {_OLLAMA_URL}: {e}. "
                f"Start with: ollama serve"
            )
        return False

    # ═══════════════════════════════════════════════════════════
    # LLM CALL DISPATCHER
    # ═══════════════════════════════════════════════════════════

    def _call_llm(self, user_prompt: str) -> Optional[dict]:
        """Route to the active backend. Returns parsed dict or None on failure."""
        if self._backend == "bedrock":
            return self._call_bedrock(user_prompt)
        elif self._backend == "anthropic":
            return self._call_anthropic(user_prompt)
        elif self._backend == "ollama":
            return self._call_ollama(user_prompt)
        return None

    # ── Backend 1: AWS Bedrock ──────────────────────────────────
    def _call_bedrock(self, user_prompt: str) -> Optional[dict]:
        """
        Call Claude on AWS Bedrock using the Messages API format.
        boto3 is already present in SageMaker -- no extra pip install needed.
        """
        if self._bedrock_client is None:
            return None
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens":        self.max_tokens,
                "system":            self._SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
            })
            response = self._bedrock_client.invoke_model(
                modelId     = _BEDROCK_MODEL,
                body        = body,
                contentType = "application/json",
                accept      = "application/json",
            )
            result = json.loads(response["body"].read())
            raw    = result["content"][0]["text"].strip()
            raw    = re.sub(r"^```(?:json)?\s*", "", raw)
            raw    = re.sub(r"\s*```$",           "", raw)
            return json.loads(raw)
        except Exception as e:
            self.logger.warning(f"[{self.name}] Bedrock call failed: {e}")
            return None

    # ── Backend 2: Anthropic direct API ─────────────────────────
    def _call_anthropic(self, user_prompt: str) -> Optional[dict]:
        """
        Call Claude via the Anthropic Python SDK.
        Requires: pip install anthropic  +  ANTHROPIC_API_KEY env var.
        """
        if self._anthropic_client is None:
            return None
        try:
            response = self._anthropic_client.messages.create(
                model      = self.model,
                max_tokens = self.max_tokens,
                system     = self._SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",           "", raw)
            return json.loads(raw)
        except Exception as e:
            self.logger.warning(f"[{self.name}] Anthropic call failed: {e}")
            return None

    # ── Backend 3: Ollama local ──────────────────────────────────
    def _call_ollama(self, user_prompt: str) -> Optional[dict]:
        """
        Call a locally running Ollama model.
        Free, no API key.
        Setup: ollama pull llama3.2 && ollama serve
        """
        try:
            full_prompt = (
                f"{self._SYSTEM_PROMPT}\n\n"
                f"IMPORTANT: Reply ONLY with valid JSON. "
                f"No markdown fences, no explanation outside JSON.\n\n"
                f"{user_prompt}"
            )
            resp = requests.post(
                f"{_OLLAMA_URL}/api/generate",
                json={
                    "model":  _OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json",    # forces Ollama to output valid JSON
                    "options": {
                        "temperature": 0.1,   # low temp for deterministic JSON
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=_OLLAMA_TIMEOUT,
            )
            if resp.status_code != 200:
                self.logger.warning(
                    f"[{self.name}] Ollama HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return None
            raw = resp.json().get("response", "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",           "", raw)
            return json.loads(raw)
        except requests.exceptions.ConnectionError:
            self.logger.warning(
                f"[{self.name}] Ollama connection refused. Start with: ollama serve"
            )
            return None
        except Exception as e:
            self.logger.warning(f"[{self.name}] Ollama call failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # RULE-BASED FALLBACK
    # ═══════════════════════════════════════════════════════════

    def _rule_based_plan(
        self,
        mean_risk:  float,
        mean_p_rf:  float,
        top_wallet: str,
        to_address: str,
        n_blocked:  int,
    ) -> dict:
        """
        Fast heuristic ActionPlan when no LLM is available.
        Deterministic, zero cost, always works.
        """
        threat_type = "phishing"
        for name, test in _THREAT_HEURISTICS.items():
            if test(mean_risk, mean_p_rf):
                threat_type = name
                break

        template_map = {
            "flash_loan": "circuit_breaker",
            "reentrancy": "circuit_breaker",
            "phishing":   "address_blocklist",
        }

        return {
            "threat_type":          threat_type,
            "severity":             _score_to_severity(mean_risk),
            "recommended_template": template_map.get(threat_type, "address_blocklist"),
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
            "llm_backend":        "rule_based",
        }

    # ═══════════════════════════════════════════════════════════
    # RAG CONTEXT RETRIEVAL
    # ═══════════════════════════════════════════════════════════

    def _get_rag_context(
        self,
        risk_score: float,
        p_rf:       float,
        s_if:       float,
        decision:   str,
        wallet:     str,
        n_results:  int = 3,
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
                m   = item["metadata"]
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
    # BUILD USER PROMPT
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _build_prompt(
        batch_idx:   int,
        n_blocked:   int,
        mean_risk:   float,
        mean_p_rf:   float,
        mean_s_if:   float,
        top_wallet:  str,
        to_address:  str,
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
    # PARSE AND VALIDATE LLM RESPONSE
    # ═══════════════════════════════════════════════════════════

    def _validate_llm_result(
        self,
        llm_result:  dict,
        mean_risk:   float,
        to_addr:     str,
        top_wallet:  str,
        rag_hits:    int,
        rag_max_sim: float,
    ) -> dict:
        """
        Fill missing / invalid keys with safe defaults.
        Ensures the ActionPlan always has the complete required schema.
        """
        valid_templates  = {"circuit_breaker", "address_blocklist", "rate_limiter"}
        valid_threats    = {"flash_loan", "reentrancy", "phishing",
                            "wash_trading", "sandwich", "unknown"}
        valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

        template = llm_result.get("recommended_template", "address_blocklist")
        if template not in valid_templates:
            self.logger.warning(
                f"[{self.name}] LLM returned invalid template '{template}', "
                f"defaulting to 'address_blocklist'"
            )
            template = "address_blocklist"

        threat_type = llm_result.get("threat_type", "unknown")
        if threat_type not in valid_threats:
            threat_type = "unknown"

        severity = llm_result.get("severity", _score_to_severity(mean_risk))
        if severity not in valid_severities:
            severity = _score_to_severity(mean_risk)

        params = llm_result.get("parameters", {}) or {}

        return {
            "threat_type":          threat_type,
            "severity":             severity,
            "recommended_template": template,
            "parameters": {
                "target_address":   params.get("target_address",   to_addr),
                "threshold":        float(params.get("threshold",  mean_risk)),
                "attacker_address": params.get("attacker_address", top_wallet),
            },
            "reasoning":            llm_result.get("reasoning", "LLM decision"),
            "rag_hits":             rag_hits,
            "rag_max_similarity":   rag_max_sim,
            "llm_used":             True,
            "llm_backend":          self._backend,
        }

    # ═══════════════════════════════════════════════════════════
    # MAIN _run
    # ═══════════════════════════════════════════════════════════

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Only activates when the batch contains BLOCK decisions.
        For all-CLEAR batches: passes through with action_plan=None.
        """
        decisions      = np.asarray(msg.payload.get("decisions",      []), dtype=object)
        risk_scores    = np.asarray(msg.payload.get("risk_scores",    []), dtype=float)
        p_rf           = np.asarray(msg.payload.get("p_rf",           []), dtype=float)
        s_if           = np.asarray(msg.payload.get("s_if",           []), dtype=float)
        batch_idx      = msg.payload.get("batch_idx", 0)
        tx_meta        = msg.payload.get("tx_meta",   {})
        policy_actions = np.asarray(
            msg.payload.get("policy_actions", []), dtype=object
        )

        # Only reason about batches with BLOCK decisions
        block_mask = (
            np.asarray(policy_actions) == "BLOCK"
            if len(policy_actions) > 0
            else np.array([], dtype=bool)
        )
        n_blocked = int(np.sum(block_mask))

        if n_blocked == 0:
            self.logger.info(
                f"[{self.name}] Batch {batch_idx+1} -- no BLOCK decisions, skipping."
            )
            return AgentMessage(
                sender=self.name,
                payload={**msg.payload, "action_plan": None},
                status="ok",
            )

        # ── Summary statistics over blocked transactions ──────────
        blocked_risks = (
            risk_scores[block_mask]
            if len(risk_scores) > 0 and np.any(block_mask)
            else risk_scores
        )
        blocked_prf = (
            p_rf[block_mask]
            if len(p_rf) > 0 and np.any(block_mask)
            else p_rf
        )
        blocked_sif = (
            s_if[block_mask]
            if len(s_if) > 0 and np.any(block_mask)
            else s_if
        )

        mean_risk = float(np.mean(blocked_risks)) if len(blocked_risks) > 0 else 0.5
        mean_p_rf = float(np.mean(blocked_prf))   if len(blocked_prf)  > 0 else 0.5
        mean_s_if = float(np.mean(blocked_sif))   if len(blocked_sif)  > 0 else 0.5

        # ── Highest-risk wallet address ───────────────────────────
        from_addrs = tx_meta.get("from_address", []) if tx_meta else []
        to_addrs   = tx_meta.get("to_address",   []) if tx_meta else []
        if len(risk_scores) > 0 and len(from_addrs) > 0:
            top_idx    = int(np.argmax(risk_scores))
            top_wallet = str(from_addrs[top_idx]) if top_idx < len(from_addrs) else "unknown"
            to_addr    = str(to_addrs[top_idx])   if top_idx < len(to_addrs)   else "unknown"
        else:
            top_wallet = "unknown"
            to_addr    = "unknown"

        # ── RAG context retrieval ─────────────────────────────────
        rag_ctx, rag_hits, rag_max_sim = self._get_rag_context(
            risk_score=mean_risk, p_rf=mean_p_rf, s_if=mean_s_if,
            decision="AUTO-BLOCK", wallet=top_wallet,
        )

        # ── LLM call (or rule-based fallback) ─────────────────────
        action_plan: dict

        if self._backend != "rule_based":
            prompt     = self._build_prompt(
                batch_idx, n_blocked, mean_risk, mean_p_rf, mean_s_if,
                top_wallet, to_addr, rag_ctx,
            )
            llm_result = self._call_llm(prompt)

            if llm_result and isinstance(llm_result, dict):
                action_plan = self._validate_llm_result(
                    llm_result, mean_risk, to_addr, top_wallet,
                    rag_hits, rag_max_sim,
                )
            else:
                # LLM returned bad/empty JSON -- rule-based fallback
                self.logger.warning(
                    f"[{self.name}] {self._backend} returned invalid JSON. "
                    f"Falling back to rule-based."
                )
                action_plan = self._rule_based_plan(
                    mean_risk, mean_p_rf, top_wallet, to_addr, n_blocked
                )
                action_plan["rag_hits"]          = rag_hits
                action_plan["rag_max_similarity"] = rag_max_sim
        else:
            # No LLM backend -- pure rule-based
            action_plan = self._rule_based_plan(
                mean_risk, mean_p_rf, top_wallet, to_addr, n_blocked
            )
            action_plan["rag_hits"]          = rag_hits
            action_plan["rag_max_similarity"] = rag_max_sim

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} | "
            f"backend={action_plan.get('llm_backend', 'rule_based')} | "
            f"threat={action_plan['threat_type']} | "
            f"severity={action_plan['severity']} | "
            f"template={action_plan['recommended_template']} | "
            f"n_blocked={n_blocked} | "
            f"rag_hits={action_plan['rag_hits']} | "
            f"llm_used={action_plan['llm_used']}"
        )

        return AgentMessage(
            sender=self.name,
            payload={**msg.payload, "action_plan": action_plan},
            status="ok",
        )

    # ═══════════════════════════════════════════════════════════
    # PUBLIC UTILITY
    # ═══════════════════════════════════════════════════════════

    @property
    def backend_name(self) -> str:
        """Return the active LLM backend name for logging / reporting."""
        return self._backend

    def backend_info(self) -> dict:
        """Return a dict summarising the active backend (useful for ExperimentTracker)."""
        info = {"backend": self._backend}
        if self._backend == "bedrock":
            info.update({"model": _BEDROCK_MODEL, "region": _BEDROCK_REGION})
        elif self._backend == "anthropic":
            info.update({"model": self.model})
        elif self._backend == "ollama":
            info.update({"model": _OLLAMA_MODEL, "url": _OLLAMA_URL})
        return info