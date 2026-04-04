import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.fraud_knowledge_agent import FraudKnowledgeAgent
from agents.decision_agent        import DecisionAgent
from agents.contract_agent        import ContractAgent
from agents.governance_agent      import GovernanceAgent
from evaluation.scenario_runner   import ScenarioRunner

RUN_DIR = "runs/run_seed42_v1"

ka = FraudKnowledgeAgent(run_dir=RUN_DIR)
da = DecisionAgent(knowledge_agent=ka, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
ca = ContractAgent(run_dir=RUN_DIR, knowledge_agent=ka)
ga = GovernanceAgent(run_dir=RUN_DIR)

print("=== CLOUD NODE ===")
cloud = ScenarioRunner(knowledge_agent=ka, decision_agent=da,
                       contract_agent=ca, governance_agent=ga,
                       node_mode="cloud", run_dir=RUN_DIR)
cloud.print_report(cloud.run_all())

print("\n=== EDGE NODE ===")
edge = ScenarioRunner(knowledge_agent=ka, decision_agent=da,
                      contract_agent=ca, governance_agent=ga,
                      node_mode="edge", run_dir=RUN_DIR)
edge.print_report(edge.run_all())