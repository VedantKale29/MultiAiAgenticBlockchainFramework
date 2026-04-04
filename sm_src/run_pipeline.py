"""
run_pipeline.py
================
UNIFIED SINGLE-COMMAND PIPELINE RUNNER

Runs the COMPLETE system from start to finish with one command:

    python run_pipeline.py                    # full run, seed=42
    python run_pipeline.py --seed 42          # explicit seed
    python run_pipeline.py --eval             # + Stage 5 evaluation scenarios
    python run_pipeline.py --eval --verbose   # + detailed per-agent logging

WHAT IT RUNS IN ORDER:
  Phase 0 - Setup        : load dataset, train RF + IF, create run dir
  Phase 1 - Base pipeline: Perception -> RF -> IF -> Fusion -> Action ->
                           Policy -> Response -> Monitoring -> Adaptation
             (all 5 batches, full metrics per batch)
  Phase 2 - Stage 1      : FraudKnowledgeAgent RAG indexing per batch
  Phase 3 - Stage 2      : AuditAgent hash-chain log + RAG self-improvement
  Phase 4 - Stage 3      : DecisionAgent (LLM/fallback) + ContractAgent
                           (RAG template selection + Slither + deploy)
  Phase 5 - Stage 4      : GovernanceAgent (rolling window -> timelock proposal)
  Phase 6 - Evaluation   : ScenarioRunner (3 scenarios, cloud + edge node)
             [only with --eval flag]

OUTPUT FILES (all in runs/run_{seed}/):
  batch_history.csv         - per-batch metrics (precision, recall, F1, etc.)
  final_state.json          - final tau, w, overall metrics
  rf_feature_importance.csv - RF feature importance
  audit_log.jsonl           - append-only hash-chain audit log
  contract_deployments.json - all autonomously deployed contracts
  governance_proposals.json - all on-chain governance proposals
  rag_store/                - ChromaDB vector store (fraud_events + templates)
  evaluation_cloud_*.json   - Stage 5 evaluation results (if --eval)

EQUIVALENT TO sm_launcher but runs LOCALLY (no AWS needed).
For AWS: python aws/sm_launcher.py --seed 42 --run_mode AGENTIC --wait
"""

import os
import sys
import json
import time
import logging
import argparse

# Force UTF-8 stdout/stderr on Windows (prevents cp1252 UnicodeEncodeError)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# -- Project root -----------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# -- Logging ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RunPipeline")

SEP  = "=" * 72
SEP2 = "-" * 72


# ==============================================================
# ARGS
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full agentic fraud detection pipeline end-to-end."
    )
    p.add_argument("--seed",     type=int,  default=42,
                   help="Random seed (default: 42, paper's exact seed)")
    p.add_argument("--eval",     action="store_true",
                   help="Run Stage 5 evaluation scenarios after main pipeline")
    p.add_argument("--verbose",  action="store_true",
                   help="Print detailed per-agent payload info")
    p.add_argument("--hardhat",  type=str,  default="http://127.0.0.1:8545",
                   help="Hardhat RPC URL (default: http://127.0.0.1:8545)")
    p.add_argument("--no_stage_agents", action="store_true",
                   help="Run base pipeline only (skip Stages 1-4)")
    return p.parse_args()


# ==============================================================
# PHASE HEADER PRINTER
# ==============================================================

def phase(title: str, subtitle: str = ""):
    log.info(SEP)
    log.info(f"  {title}")
    if subtitle:
        log.info(f"  {subtitle}")
    log.info(SEP)


# ==============================================================
# MAIN
# ==============================================================

def main():
    args  = parse_args()
    seed  = args.seed
    t_run = time.perf_counter()

    print()
    print(SEP)
    print("  AGENTIC AI BLOCKCHAIN FRAUD DETECTION")
    print("  Full Pipeline Run - Single Command")
    print(SEP)
    print(f"  seed={seed}  eval={args.eval}  hardhat={args.hardhat}")
    print(f"  stage_agents={'disabled' if args.no_stage_agents else 'enabled'}")
    print(SEP)
    print()

    # ========================================================
    # PHASE 0 - SETUP
    # ========================================================
    phase("PHASE 0 - Dataset + Model Training",
          "Load CSV -> clean -> 75:25 split -> train RF(250) + IF(200)")

    import pandas as pd
    import config as config
    from data_loader        import load_and_clean_data, get_train_test_split
    from rf_model           import RFModel
    from if_model           import IFModel
    from aws.s3_manager         import S3Manager
    from aws.cloudwatch_logger  import CloudWatchLogger
    from aws.experiment_tracker import ExperimentTracker
    from agents.coordinator_agent import CoordinatorAgent
    from agents.base_agent        import AgentMessage

    # Dataset
    dataset_path = config.get_dataset_path()
    log.info(f"Dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        log.error(f"Dataset not found: {dataset_path}")
        log.error("Place transaction_dataset.csv in the project root or set DATASET_PATH.")
        sys.exit(1)

    raw_df = pd.read_csv(dataset_path)
    log.info(f"Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")

    # Metadata columns
    label_col = "FLAG"
    wallet_col = "Address"
    df_meta = pd.DataFrame(index=raw_df.index)
    df_meta["from_address"] = raw_df[wallet_col].astype(str)
    for col in ["Index", "Unnamed: 0"]:
        if col in raw_df.columns:
            df_meta["tx_hash"] = raw_df[col].apply(lambda x: f"tx_{x}").astype(str)
            break
    else:
        df_meta["tx_hash"] = [f"tx_{i}" for i in range(len(raw_df))]
    df_meta["to_address"] = "unknown_to"
    df_meta["timestamp"]  = "not_available"

    df = load_and_clean_data(dataset_path)

    # Run directory
    run_name = config.make_run_name(seed)
    run_dir  = os.path.join(config.LOCAL_RESULTS_DIR, f"run_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run name: {run_name}")
    log.info(f"Output dir: {run_dir}")

    # AWS components (graceful fallback locally)
    s3        = S3Manager()
    cw_logger = CloudWatchLogger(run_name=run_name)
    tracker   = ExperimentTracker(run_name=run_name, run_dir=run_dir, seed=seed)

    tracker.log_params({
        "seed": seed, "batch_size": config.BATCH_SIZE,
        "rf_trees": config.RF_N_ESTIMATORS, "if_trees": config.IF_N_ESTIMATORS,
        "initial_w": config.INITIAL_WEIGHT_W0, "initial_tau": config.INITIAL_THRESHOLD_TAU0,
    })

    # Train models
    X_train, X_test, y_train, y_test = get_train_test_split(df, seed=seed)
    X_test_meta = df_meta.loc[X_test.index].copy()

    log.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
             f"| Fraud rate: {y_train.mean():.2%}")

    t0 = time.perf_counter()
    rf_model = RFModel(seed=seed)
    rf_model.train(X_train, y_train)
    rf_time = (time.perf_counter() - t0) * 1000
    rf_model.log_feature_importance(X_train,
                                    output_path=os.path.join(run_dir, "rf_feature_importance.csv"))
    log.info(f"RF trained: {rf_time:.0f}ms | features={len(X_train.columns)}")

    t0 = time.perf_counter()
    if_model = IFModel(seed=seed, y_train=y_train)
    if_model.train(X_train, y_train)
    if_time = (time.perf_counter() - t0) * 1000
    log.info(f"IF trained: {if_time:.0f}ms")

    # ========================================================
    # PHASE 1-5 - COORDINATOR (base + all stage agents)
    # ========================================================
    phase("PHASE 1–5 - Multi-Agent Pipeline",
          "Base(10 agents) + Stage1 RAG + Stage2 Audit + Stage3 Decision+Contract + Stage4 Governance")

    log.info("Building CoordinatorAgent...")
    log.info("  Base agents:  Perception -> RF -> IF -> Fusion -> Action -> Policy -> Response -> Monitoring -> Adaptation")

    if not args.no_stage_agents:
        log.info("  Stage agents: FraudKnowledgeAgent -> AuditAgent -> DecisionAgent -> ContractAgent -> GovernanceAgent")
    else:
        log.info("  Stage agents: DISABLED (--no_stage_agents flag set)")

    coordinator = CoordinatorAgent(
        rf_model          = rf_model,
        if_model          = if_model,
        expected_features = list(X_train.columns),
        run_dir           = run_dir,
        run_name          = run_name,
        seed              = seed,
        cw_logger         = cw_logger,
        tracker           = tracker,
        s3                = s3,
        hardhat_url       = args.hardhat,
        # Read contract addresses from env vars set after deploy.js
        registry_address   = os.getenv("CONTRACT_REGISTRY_ADDRESS"),
        governance_address = os.getenv("GOVERNANCE_CONTRACT_ADDRESS"),
        deployer_key       = os.getenv("HARDHAT_DEPLOYER_KEY"),
        anthropic_api_key  = os.getenv("ANTHROPIC_API_KEY"),
    )

    # Log which stage agents are active
    stage_status = {
        "FraudKnowledgeAgent": coordinator.knowledge_agent is not None,
        "AuditAgent":          coordinator.audit_agent     is not None,
        "DecisionAgent":       getattr(coordinator, "decision_agent",  None) is not None,
        "ContractAgent":       getattr(coordinator, "contract_agent",  None) is not None,
        "GovernanceAgent":     getattr(coordinator, "governance_agent",None) is not None,
    }
    log.info("")
    log.info("  Agent status:")
    for name, active in stage_status.items():
        icon = "[OK]" if active else "[WARN]  not loaded (check imports)"
        log.info(f"    {icon}  {name}")
    log.info("")

    # RUN
    log.info("Starting streaming batch pipeline...")
    t_pipeline = time.perf_counter()

    result = coordinator.run(
        AgentMessage(
            sender="RunPipeline",
            payload={
                "X_test":      X_test,
                "y_test":      y_test,
                "X_test_meta": X_test_meta,
            },
        )
    )

    pipeline_ms = (time.perf_counter() - t_pipeline) * 1000

    if result.status == "error":
        log.error(f"Pipeline failed: {result.error}")
        sys.exit(1)

    # ========================================================
    # RESULTS SUMMARY
    # ========================================================
    phase("PIPELINE RESULTS", f"Completed in {pipeline_ms/1000:.1f}s")

    history = result.payload.get("history", [])
    final   = result.payload.get("final_metrics", {})

    # Per-batch table
    print()
    print(f"{'Batch':>5} | {'w':>4} | {'tau_alert':>9} | {'Precision':>9} | {'Recall':>7} | {'F1':>7} | {'ROC-AUC':>7} | {'TP':>4} | {'FP':>3} | {'FN':>4}")
    print(SEP2)
    for row in history:
        print(
            f"{row['batch']:>5} | {row['w']:>4.2f} | "
            f"{row['tau_alert']:>9.3f} | {row['precision']:>9.3f} | "
            f"{row['recall']:>7.3f} | {row['f1']:>7.3f} | "
            f"{row.get('roc_auc', 0):>7.3f} | "
            f"{row['tp']:>4} | {row['fp']:>3} | {row['fn']:>4}"
        )
    print(SEP2)

    if final:
        print(f"\n  Mean  Precision : {final.get('mean_precision', 0):.4f}")
        print(f"  Mean  Recall    : {final.get('mean_recall',    0):.4f}")
        print(f"  Mean  F1        : {final.get('mean_f1',        0):.4f}")
        print(f"  Final tau_alert : {final.get('final_tau_alert',0):.3f}")
        print(f"  Final w         : {final.get('final_w',        0):.2f}")

    # Stage agent outputs
    print()
    log.info("Stage agent outputs:")

    # Audit log
    audit_path = os.path.join(run_dir, "audit_log.jsonl")
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            n_records = sum(1 for _ in f)
        log.info(f"  [OK] audit_log.jsonl  - {n_records} records  ({audit_path})")
    else:
        log.info(f"  [WARN]  audit_log.jsonl  - not found (AuditAgent not loaded?)")

    # Contract deployments
    dep_path = os.path.join(run_dir, "contract_deployments.json")
    if os.path.exists(dep_path):
        deployments = json.load(open(dep_path))
        log.info(f"  [OK] contract_deployments.json - {len(deployments)} deployments")
        for d in deployments:
            log.info(f"       batch={d.get('batch')} template={d.get('template')} "
                     f"incident={d.get('incident_id')} simulated={d.get('simulated')}")
    else:
        log.info("  [WARN]  contract_deployments.json - not found (ContractAgent not loaded?)")

    # Governance proposals
    gov_path = os.path.join(run_dir, "governance_proposals.json")
    if os.path.exists(gov_path):
        proposals = json.load(open(gov_path))
        log.info(f"  [OK] governance_proposals.json - {len(proposals)} proposals")
        for p in proposals:
            log.info(f"       param={p.get('param')} "
                     f"{p.get('old_value',0):.4f}->{p.get('new_value',0):.4f} "
                     f"simulated={p.get('simulated')}")
    else:
        log.info("  [WARN]  governance_proposals.json - not found (GovernanceAgent not loaded?)")

    # RAG store
    rag_dir = os.path.join(run_dir, "rag_store")
    if os.path.exists(rag_dir) and coordinator.knowledge_agent:
        sizes = coordinator.knowledge_agent.get_all_store_sizes()
        log.info(f"  [OK] RAG store - fraud_events={sizes['fraud_events']} "
                 f"contract_templates={sizes['contract_templates']}")
    else:
        log.info("  [WARN]  RAG store - not found (FraudKnowledgeAgent not loaded?)")

    # Output files
    print()
    log.info("Output files:")
    for fname in ["batch_history.csv", "final_state.json", "rf_feature_importance.csv",
                  "config.json", "audit_log.jsonl", "contract_deployments.json",
                  "governance_proposals.json"]:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            log.info(f"  [OK] {fname:<35} ({size:,} bytes)")
        else:
            log.info(f"  [WARN]  {fname:<35} not written")

    # ========================================================
    # PHASE 6 - STAGE 5 EVALUATION (optional --eval flag)
    # ========================================================
    if args.eval:
        phase("PHASE 6 - Stage 5 Evaluation Scenarios",
              "flash_loan + reentrancy + novel_variant × cloud + edge node")

        try:
            from evaluation.scenario_runner import ScenarioRunner

            eval_dir = os.path.join(run_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)

            runner = ScenarioRunner(
                knowledge_agent  = coordinator.knowledge_agent,
                decision_agent   = getattr(coordinator, "decision_agent",  None),
                contract_agent   = getattr(coordinator, "contract_agent",  None),
                governance_agent = getattr(coordinator, "governance_agent",None),
                node_mode        = "cloud",
                run_dir          = eval_dir,
            )

            # Cloud node run
            log.info("Running cloud node scenarios...")
            cloud_results = runner.run_all()
            runner.print_report(cloud_results)

            # Edge node run
            log.info("Running edge node scenarios (simulated IoT constraint)...")
            runner_edge = ScenarioRunner(
                knowledge_agent  = coordinator.knowledge_agent,
                decision_agent   = getattr(coordinator, "decision_agent",  None),
                contract_agent   = getattr(coordinator, "contract_agent",  None),
                governance_agent = getattr(coordinator, "governance_agent",None),
                node_mode        = "edge",
                run_dir          = eval_dir,
            )
            edge_results = runner_edge.run_all()
            runner_edge.print_report(edge_results)

            # Comparison table
            print()
            print(f"{'Scenario':<20} {'Cloud(ms)':>10} {'Edge(ms)':>10} {'Overhead%':>10}")
            print(SEP2[:52])
            for c, e in zip(cloud_results, edge_results):
                cloud_t = c.get("total_latency_ms") or 0
                edge_t  = e.get("total_latency_ms") or 0
                overhead = ((edge_t - cloud_t) / cloud_t * 100) if cloud_t > 0 else 0
                print(f"{c['scenario']:<20} {cloud_t:>10.0f} {edge_t:>10.0f} {overhead:>9.1f}%")

            log.info(f"Evaluation results saved to: {eval_dir}")

        except ImportError:
            log.warning("evaluation/scenario_runner.py not found - skipping Stage 5 eval")
            log.warning("Copy outputs/scenario_runner.py to evaluation/scenario_runner.py")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    total_s = (time.perf_counter() - t_run)
    print()
    print(SEP)
    print(f"  RUN COMPLETE - seed={seed}")
    print(f"  Total time : {total_s:.1f}s")
    print(f"  Output dir : {run_dir}")
    print(SEP)
    print()

    log.info("Done.")


if __name__ == "__main__":
    main()