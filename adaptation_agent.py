"""
adaptation_agent.py
===================
AGENT 7: AdaptationAgent

WHAT IS ADAPTATION IN THE PAPER?
-----------------------------------
The paper's Algorithm 2 says:

  THRESHOLD UPDATE:
    if Rec < TARGET_REC  ->> τ = max(0, τ - η_τ)      [lower threshold ->> catch more fraud]
    elif Prec < TARGET_PREC ->> τ = min(1, τ + η_τ)   [raise threshold ->> reduce false alarms]
    else                    ->> τ unchanged             [both targets met]
    τ_block = clip(τ + Δ, 0, 1)

  WEIGHT UPDATE:
    if FN > TP AND mean(s_IF_TP) > mean(p_RF_TP):
      w = clip(w - η_w, 0, 1)   [IF is explaining TPs better ->> trust IF more]
    elif FP > TP:
      w = clip(w + η_w, 0, 1)   [too many false alarms ->> trust RF more]
    else:
      w unchanged

WHAT DOES THIS AGENT DO?
-------------------------
AdaptationAgent receives the monitoring report and:
  1. Reads current tau, w from agent_state
  2. Applies the threshold update rule from Algorithm 2
  3. Applies the weight update rule from Algorithm 2
  4. Returns the NEW state (new_tau, new_tau_block, new_w)

The CoordinatorAgent then UPDATES the shared agent state dictionary
so that FusionAgent uses the new values in the next batch.

WHY IS THIS THE MOST IMPORTANT AGENT?
---------------------------------------
This is what makes the system AGENTIC rather than just a model.
Static ML models use FIXED thresholds. This agent LEARNS from each
batch and adjusts the decision boundaries automatically.

Without AdaptationAgent:
  - Recall stays at ~0.71 (whatever batch 1 gives)
  - No improvement over time

With AdaptationAgent:
  - Recall improves from ~0.71 ->> ~0.84 across batches
  - This is exactly what Table 3 and Table 5 in the paper show

INPUT  (AgentMessage payload):
  - "p_rf_tp"     : np.ndarray > RF scores for True Positives
  - "s_if_tp"     : np.ndarray — IF scores for True Positives
  - "tp"          : int        — True Positive count
  - "fp"          : int        — False Positive count
  - "fn"          : int        — False Negative count
  - "prec"        : float      — batch precision
  - "rec"         : float      — batch recall
  - "batch_idx"   : int        — batch number
  - "agent_state" : dict       — current {w, tau_alert, tau_block}

OUTPUT (AgentMessage payload):
  - "new_state"   : dict       — updated {w, tau_alert, tau_block}
  - "batch_idx"   : int        — passed through
"""

import numpy as np

from  base_agent import BaseAgent, AgentMessage
import config


class AdaptationAgent(BaseAgent):

    def __init__(self):
        super().__init__(name="AdaptationAgent")
        self.target_prec = config.TARGET_PRECISION
        self.target_rec  = config.TARGET_RECALL
        self.eta_tau     = config.STEP_SIZE_TAU
        self.eta_w       = config.STEP_SIZE_W
        self.delta       = config.BLOCK_MARGIN_DELTA

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Apply Algorithm 2: update tau and w based on batch performance.
        """
        p_rf_tp: np.ndarray = msg.payload["p_rf_tp"]
        s_if_tp: np.ndarray = msg.payload["s_if_tp"]
        tp: int             = msg.payload["tp"]
        fp: int             = msg.payload["fp"]
        fn: int             = msg.payload["fn"]
        prec: float         = msg.payload["prec"]
        rec: float          = msg.payload["rec"]
        batch_idx: int      = msg.payload["batch_idx"]
        agent_state: dict   = msg.payload["agent_state"]

        tau = agent_state["tau_alert"]
        w   = agent_state["w"]

        # ═══════════════════════════════════════════════════════════
        # PART 1: THRESHOLD UPDATE (Algorithm 2, lines 2-6)
        # ═══════════════════════════════════════════════════════════
        new_tau = tau

        if rec < self.target_rec:
            # Recall too low ->> lower threshold ->> flag MORE transactions
            new_tau = max(0.0, tau - self.eta_tau)
            self.logger.info(
                f"[{self.name}] Recall {rec:.3f} < {self.target_rec} "
                f"->> LOWER tau: {tau:.3f} ->> {new_tau:.3f}"
            )

        elif prec < self.target_prec:
            # Precision too low ->> raise threshold ->> flag FEWER transactions
            new_tau = min(1.0, tau + self.eta_tau)
            self.logger.info(
                f"[{self.name}] Precision {prec:.3f} < {self.target_prec} "
                f"->> RAISE tau: {tau:.3f} ->> {new_tau:.3f}"
            )

        else:
            self.logger.info(
                f"[{self.name}] Both targets met ->> tau unchanged at {tau:.3f}"
            )

        # Block threshold is always alert threshold + delta
        new_tau_block = float(np.clip(new_tau + self.delta, 0.0, 1.0))

        # ═══════════════════════════════════════════════════════════
        # PART 2: WEIGHT UPDATE (Algorithm 2, lines 8-14)
        # ═══════════════════════════════════════════════════════════
        new_w = w

        if tp == 0:
            self.logger.info(
                f"[{self.name}] No TP in batch > weight unchanged at w={w:.2f}"
            )
        else:
            mean_p_rf_tp = float(np.mean(p_rf_tp)) if len(p_rf_tp) > 0 else 0.0
            mean_s_if_tp = float(np.mean(s_if_tp)) if len(s_if_tp) > 0 else 0.0

            if fn > tp and mean_s_if_tp > mean_p_rf_tp:
                # IF is explaining true positives better than RF
                # ->> shift weight toward IF (decrease w)
                new_w = float(np.clip(w - self.eta_w, 0.0, 1.0))
                self.logger.info(
                    f"[{self.name}] FN({fn}) > TP({tp}) AND IF_avg({mean_s_if_tp:.3f}) "
                    f"> RF_avg({mean_p_rf_tp:.3f}) ->> DECREASE w: {w:.2f} ->> {new_w:.2f}"
                )

            elif fp > tp:
                # Too many false positives ->> RF is over-flagging
                # ->> trust RF MORE (increase w, since RF is more precise)
                new_w = float(np.clip(w + self.eta_w, 0.0, 1.0))
                self.logger.info(
                    f"[{self.name}] FP({fp}) > TP({tp}) "
                    f"->> INCREASE w: {w:.2f} ->> {new_w:.2f}"
                )

            else:
                self.logger.info(
                    f"[{self.name}] Weight conditions not met > w unchanged at {w:.2f}"
                )

        # ── Build new state ────────────────────────────────────────
        new_state = {
            "w"          : new_w,
            "tau_alert"  : new_tau,
            "tau_block"  : new_tau_block,
        }

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} Adaptation complete ->> "
            f"w={new_w:.2f} tau_alert={new_tau:.3f} tau_block={new_tau_block:.3f}"
        )

        return AgentMessage(
            sender=self.name,
            payload={
                "new_state" : new_state,
                "batch_idx" : batch_idx,
            },
            status="ok",
        )
