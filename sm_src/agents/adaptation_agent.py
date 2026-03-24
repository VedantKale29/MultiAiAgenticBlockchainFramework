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
      w = clip(w + η_w, 0, 1)   [too many false alarms ->> trust RF more] where η_w =
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
 
from agents.base_agent import BaseAgent, AgentMessage
import config
 
 
class AdaptationAgent(BaseAgent):
 
    def __init__(self, cw_logger=None):
        super().__init__(name="AdaptationAgent")
        self.cw_logger   = cw_logger
        self.target_prec = config.TARGET_PRECISION
        self.target_rec  = config.TARGET_RECALL
        self.delta       = config.BLOCK_MARGIN_DELTA
 
        # ── PI gains for tau ───────────────────────────────────────
        # v2: K_p raised to 0.30 from 0.10.
        # Derivation: K_p ≈ old_fixed_step / typical_gap
        #             = 0.02 / 0.06 ≈ 0.33 → use 0.30 (conservative)
        self.K_p_tau = getattr(config, "PI_KP_TAU", 0.30)
        self.K_i_tau = getattr(config, "PI_KI_TAU", 0.02)
 
        # ── PI gains for weight ────────────────────────────────────
        self.K_p_w   = getattr(config, "PI_KP_W",   0.15)
        self.K_i_w   = getattr(config, "PI_KI_W",   0.02)
 
        # ── Anti-windup ────────────────────────────────────────────
        self.max_integral = getattr(config, "PI_MAX_INTEGRAL", 1.0)
 
        # ── Tau step bounds ────────────────────────────────────────
        # min raised to 0.015 so controller never stalls near target
        self.min_tau_step = getattr(config, "PI_MIN_TAU_STEP", 0.015)
        self.max_tau_step = getattr(config, "PI_MAX_TAU_STEP", 0.10)
 
        # ── Weight movement threshold ──────────────────────────────
        # lowered to 0.03: w moves when gap difference > 3% (not 5%)
        self.weight_move_threshold = getattr(config, "PI_WEIGHT_MOVE_THRESHOLD", 0.03)
 
        # ── Recall priority bias ───────────────────────────────────
        # Encodes the asymmetric cost: FN (missed fraud) > FP (false alarm)
        # rec_gap_w = RECALL_WEIGHT × rec_gap  →  bigger step for recall
        # prec_gap_w = PREC_WEIGHT  × prec_gap →  normal step for precision
        # Set both to 1.0 for symmetric (paper-equivalent) behaviour.
        self.recall_weight = getattr(config, "PI_RECALL_WEIGHT", 1.5)
        self.prec_weight   = getattr(config, "PI_PREC_WEIGHT",   1.0)
 
    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════
    @staticmethod
    def _f1(p: float, r: float) -> float:
        """F1 score from precision and recall."""
        return 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0
 
    def _update_integral(self, current: float, error: float) -> float:
        """Accumulate error into integral with anti-windup clamping."""
        return float(np.clip(current + error, -self.max_integral, self.max_integral))
 
    def _pi_step(
        self,
        K_p: float,
        K_i: float,
        error_magnitude: float,
        integral: float,
        min_step: float = 0.0,
        max_step: float = 1.0,
    ) -> float:
        """Compute PI step size: proportional + integral, clamped to [min, max]."""
        raw = K_p * error_magnitude + K_i * abs(integral)
        return float(np.clip(raw, min_step, max_step))
 
    # ══════════════════════════════════════════════════════════════
    # MAIN RUN
    # ══════════════════════════════════════════════════════════════
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
 
        old_tau      = agent_state["tau_alert"]
        old_w        = agent_state["w"]
        tau_integral = float(agent_state.get("tau_integral", 0.0))
        w_integral   = float(agent_state.get("w_integral",   0.0))
 
        # ════════════════════════════════════════════════════════
        # STEP 1 — Raw gaps and weighted gaps
        # ════════════════════════════════════════════════════════
        rec_gap  = max(0.0, self.target_rec  - rec)
        prec_gap = max(0.0, self.target_prec - prec)
 
        # Weighted gaps: recall amplified by RECALL_WEIGHT
        rec_gap_w  = self.recall_weight * rec_gap
        prec_gap_w = self.prec_weight   * prec_gap
 
        f1_target  = self._f1(self.target_prec, self.target_rec)
        f1_current = self._f1(prec, rec)
        f1_gap     = max(0.0, f1_target - f1_current)
 
        recall_below    = rec  < self.target_rec
        precision_below = prec < self.target_prec
 
        # ════════════════════════════════════════════════════════
        # STEP 2 — TAU UPDATE (Generalised PI with recall bias)
        # ════════════════════════════════════════════════════════
        #
        # e_tau sign convention:
        #   e_tau > 0  →  tau should DROP  (catch more fraud)
        #   e_tau < 0  →  tau should RISE  (cut false alarms)
        #   e_tau = 0  →  both targets met
        #
        # Uses WEIGHTED gaps so recall deficiency always produces
        # a larger correction than an equal precision deficiency.
 
        tau_case = "both_met"
        e_tau    = 0.0
 
        if recall_below and precision_below:
            # Both deficient — use weighted imbalance to pick direction.
            # If equally weighted-deficient, recall wins (fraud safety).
            imbalance = rec_gap_w - prec_gap_w
            if imbalance >= 0:                   # recall at least as bad
                e_tau = f1_gap                   # positive → lower tau
            else:
                e_tau = -f1_gap                  # negative → raise tau
            tau_case = "both_below"
 
        elif recall_below:
            # Only recall deficient — use weighted gap for larger step
            e_tau    = rec_gap_w
            tau_case = "recall_below"
 
        elif precision_below:
            # Only precision deficient — use weighted gap
            e_tau    = -prec_gap_w
            tau_case = "precision_below"
 
        # Update integral accumulator
        tau_integral = self._update_integral(tau_integral, e_tau)
 
        # Compute dynamic step with floor and ceiling
        eta_tau = self._pi_step(
            self.K_p_tau,
            self.K_i_tau,
            abs(e_tau),
            tau_integral,
            min_step=self.min_tau_step if e_tau != 0.0 else 0.0,
            max_step=self.max_tau_step,
        )
 
        new_tau   = old_tau
        tau_event = "threshold_unchanged"
        tau_reason = tau_case
 
        if e_tau > 1e-6:
            new_tau   = float(np.clip(old_tau - eta_tau, 0.0, 1.0))
            tau_event = "threshold_lowered"
            tau_reason = (
                f"{tau_case} | rec={rec:.3f} raw_gap={rec_gap:.3f} "
                f"w_gap={rec_gap_w:.3f} | e={e_tau:.4f} step={eta_tau:.4f}"
            )
            self.logger.info(
                f"[{self.name}] TAU ↓  [{tau_case}] "
                f"R={rec:.3f}(gap={rec_gap:.3f}×{self.recall_weight}→{rec_gap_w:.3f}) "
                f"P={prec:.3f}(gap={prec_gap:.3f}) "
                f"e={e_tau:.4f} η={eta_tau:.4f} "
                f"{old_tau:.4f}→{new_tau:.4f}"
            )
 
        elif e_tau < -1e-6:
            new_tau   = float(np.clip(old_tau + eta_tau, 0.0, 1.0))
            tau_event = "threshold_raised"
            tau_reason = (
                f"{tau_case} | prec={prec:.3f} raw_gap={prec_gap:.3f} "
                f"w_gap={prec_gap_w:.3f} | e={e_tau:.4f} step={eta_tau:.4f}"
            )
            self.logger.info(
                f"[{self.name}] TAU ↑  [{tau_case}] "
                f"R={rec:.3f}(gap={rec_gap:.3f}) "
                f"P={prec:.3f}(gap={prec_gap:.3f}×{self.prec_weight}→{prec_gap_w:.3f}) "
                f"e={e_tau:.4f} η={eta_tau:.4f} "
                f"{old_tau:.4f}→{new_tau:.4f}"
            )
 
        else:
            self.logger.info(
                f"[{self.name}] TAU — unchanged {old_tau:.4f} "
                f"(R={rec:.3f} P={prec:.3f} both targets met)"
            )
 
        new_tau_block = float(np.clip(new_tau + self.delta, 0.0, 1.0))
 
        # ════════════════════════════════════════════════════════
        # STEP 3 — WEIGHT UPDATE (Generalised PI with recall bias)
        # ════════════════════════════════════════════════════════
        #
        # e_w = prec_gap_w - rec_gap_w
        #   Positive: precision more deficient (weighted) → w up (more RF)
        #   Negative: recall more deficient (weighted)    → w down (more IF)
        #
        # Because rec_gap_w = 1.5 × rec_gap, the controller is biased
        # toward decreasing w whenever recall and precision gaps are
        # close in magnitude. This is the intended asymmetric behaviour.
 
        new_w    = old_w
        w_event  = "weight_unchanged"
        w_reason = "conditions_not_triggered"
 
        if tp == 0:
            e_w      = 0.0
            w_reason = "no_true_positives"
            self.logger.info(
                f"[{self.name}] W — no TP, weight frozen at {old_w:.3f}"
            )
        else:
            mean_rf_tp   = float(np.mean(p_rf_tp)) if len(p_rf_tp) > 0 else 0.0
            mean_if_tp   = float(np.mean(s_if_tp)) if len(s_if_tp) > 0 else 0.0
            detector_adv = mean_if_tp - mean_rf_tp
 
            # Signed error uses weighted gaps
            e_w = prec_gap_w - rec_gap_w
 
            w_integral = self._update_integral(w_integral, e_w)
 
            eta_w = self._pi_step(
                self.K_p_w,
                self.K_i_w,
                abs(e_w),
                w_integral,
                min_step=0.0,
                max_step=0.20,
            )
 
            if e_w < -self.weight_move_threshold and detector_adv > 0:
                # Recall more deficient (weighted) AND IF explains TPs better
                # → decrease w (give more weight to IF)
                new_w    = float(np.clip(old_w - eta_w, 0.0, 1.0))
                w_event  = "weight_decreased"
                w_reason = (
                    f"recall_dominant_weighted | e_w={e_w:.4f} "
                    f"IF_adv={detector_adv:.4f} step={eta_w:.4f}"
                )
                self.logger.info(
                    f"[{self.name}] W ↓  recall_dominant_weighted "
                    f"e_w={e_w:.4f} IF_adv={detector_adv:.4f} "
                    f"η={eta_w:.4f} {old_w:.3f}→{new_w:.3f}"
                )
 
            elif e_w > self.weight_move_threshold and fp > fn:
                # Precision more deficient (weighted) AND FP > FN confirms
                # → increase w (give more weight to RF)
                new_w    = float(np.clip(old_w + eta_w, 0.0, 1.0))
                w_event  = "weight_increased"
                w_reason = (
                    f"precision_dominant_weighted | e_w={e_w:.4f} "
                    f"FP({fp})>FN({fn}) step={eta_w:.4f}"
                )
                self.logger.info(
                    f"[{self.name}] W ↑  precision_dominant_weighted "
                    f"e_w={e_w:.4f} FP={fp} FN={fn} "
                    f"η={eta_w:.4f} {old_w:.3f}→{new_w:.3f}"
                )
 
            else:
                # Determine why weight didn't move (for logging clarity)
                if abs(e_w) <= self.weight_move_threshold:
                    w_reason = (
                        f"gap_within_tolerance | e_w={e_w:.4f} "
                        f"threshold={self.weight_move_threshold}"
                    )
                elif e_w < -self.weight_move_threshold and detector_adv <= 0:
                    w_reason = (
                        f"recall_dominant_but_rf_stronger | "
                        f"e_w={e_w:.4f} detector_adv={detector_adv:.4f}"
                    )
                else:
                    w_reason = (
                        f"fp_not_dominant | e_w={e_w:.4f} FP={fp} FN={fn}"
                    )
                self.logger.info(
                    f"[{self.name}] W — unchanged {old_w:.3f} ({w_reason})"
                )
 
        # ════════════════════════════════════════════════════════
        # STEP 4 — CloudWatch logging (unchanged interface)
        # ════════════════════════════════════════════════════════
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
 
        # ════════════════════════════════════════════════════════
        # STEP 5 — Build new state (backward-compatible)
        # ════════════════════════════════════════════════════════
        new_state = {
            "w"            : new_w,
            "tau_alert"    : new_tau,
            "tau_block"    : new_tau_block,
            "tau_integral" : tau_integral,
            "w_integral"   : w_integral,
        }
 
        self.logger.info(
            f"[{self.name}] Batch {batch_idx+1} summary → "
            f"tau: {old_tau:.4f}→{new_tau:.4f} (η={abs(new_tau-old_tau):.4f}) | "
            f"w: {old_w:.3f}→{new_w:.3f} | "
            f"integrals: τ={tau_integral:.4f} w={w_integral:.4f} | "
            f"recall_weight={self.recall_weight} prec_weight={self.prec_weight}"
        )
 
        return AgentMessage(
            sender=self.name,
            payload={"new_state": new_state, "batch_idx": batch_idx},
            status="ok",
        )