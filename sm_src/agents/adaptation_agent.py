"""
adaptation_agent.py
===================
AGENT 7: AdaptationAgent

WHAT IS ADAPTATION IN THE PAPER?
-----------------------------------
ROLE IN PAPER:
  "The agent self-tunes thresholds and detector weights between
   transaction batches to achieve target precision and recall
   without human feedback."

-----------------------------------------------------------------------
τ = alert threshold, w = RF weight in fusion, Δ = block margin delta i.e. how much higher block threshold is than alert threshold
TARGET_PREC = 0.75, TARGET_REC = 0.80
-----------------------------------------------------------------------
τ = alert threshold, w = RF weight in fusion, Δ = block margin delta i.e. how much higher block threshold is than alert threshold
by default, τ0 = 0.487, w0 = 0.70, Δ = 0.10

  THRESHOLD UPDATE:
    if Rec < TARGET_REC  ->> τ = max(0, τ - η_τ)      [lower threshold ->> catch more fraud]  recall = TP / (TP + FN) where η_τ = step size for tau, e.g. 0.02
    elif Prec < TARGET_PREC ->> τ = min(1, τ + η_τ)   [raise threshold ->> reduce false alarms] # precision = TP / (TP + FP)
    else                    ->> τ unchanged             [both targets met]
    τ_block = clip(τ + Δ, 0, 1)  clip means to constrain the value between 0 and 1.... Example: if τ + Δ = 1.05, then τ_block would be set to 1.0; if τ + Δ = -0.02, then τ_block would be set to 0.0

  WEIGHT UPDATE:
    if FN > TP AND mean(s_IF_TP) > mean(p_RF_TP):    where s_IF_TP and p_RF_TP are the IF and RF scores for the TRUE POSITIVES in the batch
      w = clip(w - η_w, 0, 1)   [IF is explaining TPs better ->> trust IF more]  where FN > TP means we're missing too many frauds, and if the IF scores for the TPs are higher than the RF scores, it suggests that the IF is doing a better job at identifying those frauds. So we decrease w to rely more on IF.
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

AWS INTEGRATION HERE:
  After updating tau and w, logs the adaptation event to CloudWatch.
  This lets you search CloudWatch for:
    "threshold_lowered" → see when recall was below target
    "weight_unchanged"  → see when system was stable
"""

import numpy as np

from  agents.base_agent import BaseAgent, AgentMessage
import  config as config


class AdaptationAgent(BaseAgent):

    def __init__(self, cw_logger=None):
        """
        Parameters
        ----------
        cw_logger : CloudWatchLogger | None
            Used to log adaptation events to CloudWatch.
        """
        super().__init__(name="AdaptationAgent")
        self.cw_logger    = cw_logger
        self.target_prec  = config.TARGET_PRECISION
        self.target_rec   = config.TARGET_RECALL
        self.eta_tau      = config.STEP_SIZE_TAU
        self.eta_w        = config.STEP_SIZE_W
        self.delta        = config.BLOCK_MARGIN_DELTA

    def _run(self, msg: AgentMessage) -> AgentMessage:
        p_rf_tp     = msg.payload["p_rf_tp"]
        s_if_tp     = msg.payload["s_if_tp"]
        tp          = msg.payload["tp"]
        fp          = msg.payload["fp"]
        fn          = msg.payload["fn"]
        prec        = msg.payload["prec"]
        rec         = msg.payload["rec"]
        batch_idx   = msg.payload["batch_idx"]
        agent_state = msg.payload["agent_state"]

        old_tau = agent_state["tau_alert"]
        old_w   = agent_state["w"]

        # ════════════════════════════════════════════════════════
        # THRESHOLD UPDATE (Algorithm 2, lines 2–6)
        # ════════════════════════════════════════════════════════
        new_tau    = old_tau
        tau_event  = "threshold_unchanged"
        tau_reason = "both_targets_met"

        if rec < self.target_rec:
            new_tau    = max(0.0, old_tau - self.eta_tau)                # era_tau = 
            tau_event  = "threshold_lowered"
            tau_reason = "recall_below_target"
            self.logger.info(
                f"[{self.name}] Recall {rec:.3f} < {self.target_rec} "
                f"→ LOWER tau: {old_tau:.3f} → {new_tau:.3f}"
            )
        elif prec < self.target_prec:
            new_tau    = min(1.0, old_tau + self.eta_tau)
            tau_event  = "threshold_raised"
            tau_reason = "precision_below_target"
            self.logger.info(
                f"[{self.name}] Prec {prec:.3f} < {self.target_prec} "
                f"→ RAISE tau: {old_tau:.3f} → {new_tau:.3f}"
            )
        else:
            self.logger.info(f"[{self.name}] Both targets met — tau unchanged at {old_tau:.3f}")

        new_tau_block = float(np.clip(new_tau + self.delta, 0.0, 1.0))

        # ════════════════════════════════════════════════════════
        # WEIGHT UPDATE (Algorithm 2, lines 8–14)
        # ════════════════════════════════════════════════════════
        new_w    = old_w
        w_event  = "weight_unchanged"
        w_reason = "conditions_not_triggered"

        if tp == 0:
            self.logger.info(f"[{self.name}] No TP — weight unchanged at w={old_w:.2f}")
            w_reason = "no_true_positives"
        else:
            mean_rf_tp = float(np.mean(p_rf_tp)) if len(p_rf_tp) > 0 else 0.0
            mean_if_tp = float(np.mean(s_if_tp)) if len(s_if_tp) > 0 else 0.0

            if fn > tp and mean_if_tp > mean_rf_tp:
                new_w    = float(np.clip(old_w - self.eta_w, 0.0, 1.0))
                w_event  = "weight_decreased"
                w_reason = "fn_dominated_if_stronger"
                self.logger.info(
                    f"[{self.name}] FN({fn})>TP({tp}) & IF({mean_if_tp:.3f})>RF({mean_rf_tp:.3f})"
                    f" → DECREASE w: {old_w:.2f}→{new_w:.2f}"
                )
            elif fp > tp:
                new_w    = float(np.clip(old_w + self.eta_w, 0.0, 1.0))
                w_event  = "weight_increased"
                w_reason = "fp_dominated"
                self.logger.info(
                    f"[{self.name}] FP({fp})>TP({tp}) → INCREASE w: {old_w:.2f}→{new_w:.2f}"
                )
            else:
                self.logger.info(f"[{self.name}] Weight unchanged at w={old_w:.2f}")

        # ════════════════════════════════════════════════════════
        # AWS INTEGRATION — Log adaptation event to CloudWatch
        # ════════════════════════════════════════════════════════
        # This creates a searchable record in CloudWatch each time
        # the agent changes its tau or w. You can filter for:
        #   "threshold_lowered" to see when recall was too low
        #   "weight_decreased"  to see when IF was helping more
        if self.cw_logger:
            self.cw_logger.log_adaptation(
                batch_idx = batch_idx,
                event     = f"{tau_event}+{w_event}",
                old_tau   = old_tau,
                new_tau   = new_tau,
                old_w     = old_w,
                new_w     = new_w,
                reason    = f"tau:{tau_reason} | w:{w_reason}",
            )

        new_state = {
            "w"         : new_w,
            "tau_alert" : new_tau,
            "tau_block" : new_tau_block,
        }

        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} → "
            f"w={new_w:.2f} tau={new_tau:.3f} tau_b={new_tau_block:.3f}"
        )

        return AgentMessage(
            sender=self.name,
            payload={"new_state": new_state, "batch_idx": batch_idx},
            status="ok",
        )






