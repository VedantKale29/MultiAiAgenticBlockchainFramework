"""
agents/fraud_knowledge_agent.py
================================
STAGE 1 -- RAG Knowledge Base Agent

WHAT IT DOES:
  Maintains TWO ChromaDB collections in the same vector store:

    1. "fraud_events"        -- historical fraud detections from fraud_events.csv
                               Queried by AuditAgent and DecisionAgent for
                               evidence-backed decisions (RAP pattern).

    2. "contract_templates"  -- pre-audited Solidity response templates
                               (CircuitBreaker, AddressBlocklist, RateLimiter)
                               Queried by ContractAgent to select the best-matching
                               template for a given threat type by cosine similarity.

  This closes Gap 1: the RAG-based template selection path is now fully
  exercised. ContractAgent.query_similar() returns real results from
  the vector store instead of falling through to the built-in dict.

HOW THE TWO COLLECTIONS RELATE:
  fraud_events         → evidence for WHY a decision was made
  contract_templates   → evidence for WHAT response to deploy

BOTH COLLECTIONS:
  - Are stored in the same ChromaDB PersistentClient at {run_dir}/rag_store/
  - Use the same sentence-transformer embedding model
  - Are upsert-safe (calling index_* multiple times never duplicates)
  - Degrade gracefully if chromadb/sentence-transformers not installed

ZERO BREAKING CHANGES:
  - All existing method signatures are preserved exactly
  - If chromadb not installed: both collections disabled, pipeline unchanged
  - contract_agent.py's _select_template() will now find real RAG results

DEPENDENCIES:
  pip install chromadb sentence-transformers

VECTOR STORE LOCATION:
  {run_dir}/rag_store/   -- one PersistentClient, two collections
"""

import os
import json
import csv
import textwrap
from datetime import datetime
from typing import Optional

from agents.base_agent import BaseAgent, AgentMessage


# ══════════════════════════════════════════════════════════════
# TEMPLATE DEFINITIONS
# Each template is indexed as a ChromaDB document.
# The text field is what gets embedded -- written to be semantically
# rich so cosine similarity retrieval works correctly.
# The metadata carries the key fields ContractAgent needs.
# ══════════════════════════════════════════════════════════════

CONTRACT_TEMPLATES = {
    "circuit_breaker": {
        # ── embeddable text ──────────────────────────────────
        # EMBEDDING NOTE: keyword-dense, terms repeated in multiple phrasings
        # so all-MiniLM-L6-v2 cosine similarity scores high against short queries.
        "text": (
            "circuit_breaker contract template. "
            "threat type: flash_loan flash loan attack rapid drain single block. "
            "threat type: reentrancy reentrancy attack repeated external call. "
            "threat type: price_oracle oracle manipulation outflow spike. "
            "severity CRITICAL severity HIGH. "
            "recommended template: circuit_breaker. "
            "use circuit_breaker for flash_loan CRITICAL HIGH. "
            "use circuit_breaker for reentrancy HIGH CRITICAL. "
            "use circuit_breaker for oracle manipulation CRITICAL. "
            "mechanism: calls pause on target contract halts all protocol activity. "
            "OpenZeppelin Pausable circuit breaker pattern. "
            "stops flash loan by pausing pool. stops reentrancy by pausing contract. "
            "deploy circuit_breaker when anomaly score above 0.85 rapid extraction."
        ),
        # ── metadata stored alongside the document ───────────
        "metadata": {
            "template_key":  "circuit_breaker",
            "threat_types":  "flash_loan,reentrancy,price_oracle,outflow_spike",
            "severity":      "CRITICAL,HIGH",
            "mechanism":     "pause",
            "description":   "Calls pause() on target contract to halt all activity",
            "openzeppelin":  "Pausable",
            "deploy_time_s": "fast",   # <2s Slither, <5s deploy
            "audited":       "true",
        },
        # ── standalone Solidity (also written to contracts/) ──
        "solidity": textwrap.dedent("""\
            // SPDX-License-Identifier: MIT
            pragma solidity ^0.8.20;

            /**
             * CircuitBreaker_{{INCIDENT_ID}}.sol
             * ====================================
             * Pre-audited response template -- STAGE 1 RAG contract_templates
             *
             * PURPOSE:
             *   Emergency circuit-breaker deployed autonomously by ContractAgent
             *   when a flash loan or reentrancy attack is detected.
             *   Calls pause() on the target vulnerable protocol to halt all activity.
             *
             * USE FOR:
             *   - Flash loan attacks (rapid single-block drain)
             *   - Reentrancy attacks (repeated external call exploit)
             *   - Price oracle manipulation
             *   - Any HIGH/CRITICAL anomaly requiring immediate protocol halt
             *
             * MAPS TO FRAMEWORK:
             *   Section 4.1: "autonomous contract selection and parameterisation"
             *   RO5: deployed without human intervention within 12-second window
             *
             * AUDITED WITH: Mythril (offline, pre-deployment)
             * VERIFIED WITH: Slither (hot path, <2 seconds)
             */

            interface IPausable {
                function pause() external;
            }

            contract CircuitBreaker_{{INCIDENT_ID}} {

                address public immutable owner;
                address public immutable target;
                bool    public activated;

                event CircuitBreakerActivated(
                    address indexed target,
                    address indexed activatedBy,
                    uint256 timestamp
                );

                constructor(address _target) {
                    require(_target != address(0), "Invalid target");
                    owner  = msg.sender;
                    target = _target;
                }

                modifier onlyOwner() {
                    require(msg.sender == owner, "Not owner");
                    _;
                }

                /**
                 * Activate the circuit breaker.
                 * Calls pause() on the target contract to halt all activity.
                 * Can only be called once (idempotent after activation).
                 */
                function activate() external onlyOwner {
                    require(!activated, "Already activated");
                    activated = true;
                    IPausable(target).pause();
                    emit CircuitBreakerActivated(target, msg.sender, block.timestamp);
                }

                function isActivated() external view returns (bool) {
                    return activated;
                }
            }
            """),
    },

    "address_blocklist": {
        # EMBEDDING NOTE: keyword-dense for address/wallet/phishing queries.
        "text": (
            "address_blocklist contract template. "
            "threat type: phishing phishing attack malicious wallet scam. "
            "threat type: wash_trading wash trading circular transactions. "
            "threat type: laundering wallet laundering suspicious address. "
            "threat type: malicious_wallet known attacker address. "
            "severity HIGH severity MEDIUM. "
            "recommended template: address_blocklist. "
            "use address_blocklist for phishing HIGH. "
            "use address_blocklist for wash_trading MEDIUM. "
            "use address_blocklist for malicious wallet scam address. "
            "mechanism: records attacker wallet address in mapping on chain. "
            "emits BlockedAddress event for enforcement compliance monitoring. "
            "permanently flags wallet address on chain for all systems. "
            "deploy address_blocklist when specific wallet is the threat actor."
        ),
        "metadata": {
            "template_key":  "address_blocklist",
            "threat_types":  "phishing,wash_trading,laundering,scam,malicious_wallet",
            "severity":      "HIGH,MEDIUM",
            "mechanism":     "blocklist",
            "description":   "Records attacker address; emits BlockedAddress event",
            "openzeppelin":  "none",
            "deploy_time_s": "fast",
            "audited":       "true",
        },
        "solidity": textwrap.dedent("""\
            // SPDX-License-Identifier: MIT
            pragma solidity ^0.8.20;

            /**
             * AddressBlocklist_{{INCIDENT_ID}}.sol
             * ======================================
             * Pre-audited response template -- STAGE 1 RAG contract_templates
             *
             * PURPOSE:
             *   Autonomously deployed by ContractAgent to permanently record
             *   a malicious wallet address on-chain when phishing or wash
             *   trading is detected.
             *
             * USE FOR:
             *   - Phishing attacks (known malicious wallet)
             *   - Wash trading (repeated circular transactions)
             *   - Address-level laundering patterns
             *   - Any MEDIUM/HIGH anomaly with identifiable attacker address
             *
             * MAPS TO FRAMEWORK:
             *   Section 4.1: "autonomous contract selection and parameterisation"
             *   RO5: deployed without human intervention
             *
             * AUDITED WITH: Mythril (offline, pre-deployment)
             * VERIFIED WITH: Slither (hot path, <2 seconds)
             */

            contract AddressBlocklist_{{INCIDENT_ID}} {

                address public immutable owner;

                mapping(address => bool)    public blocked;
                mapping(address => uint256) public riskScores;
                mapping(address => uint256) public blockedAt;

                event BlockedAddress(
                    address indexed wallet,
                    uint256         riskScore,
                    uint256         timestamp
                );
                event RemovedAddress(
                    address indexed wallet,
                    uint256         timestamp
                );

                constructor() {
                    owner = msg.sender;
                }

                modifier onlyOwner() {
                    require(msg.sender == owner, "Not owner");
                    _;
                }

                /**
                 * Block an address. riskScore is scaled x1e18 (e.g. 0.95 = 950000000000000000).
                 * Safe to call multiple times -- idempotent after first block.
                 */
                function blockAddress(
                    address wallet,
                    uint256 riskScore
                ) external onlyOwner {
                    require(wallet != address(0), "Invalid address");
                    blocked[wallet]    = true;
                    riskScores[wallet] = riskScore;
                    blockedAt[wallet]  = block.timestamp;
                    emit BlockedAddress(wallet, riskScore, block.timestamp);
                }

                function removeAddress(address wallet) external onlyOwner {
                    blocked[wallet] = false;
                    emit RemovedAddress(wallet, block.timestamp);
                }

                function isBlocked(address wallet) external view returns (bool) {
                    return blocked[wallet];
                }

                function getRiskScore(address wallet) external view returns (uint256) {
                    return riskScores[wallet];
                }
            }
            """),
    },

    "rate_limiter": {
        # EMBEDDING NOTE: keyword-dense for volume/sandwich/novel queries.
        "text": (
            "rate_limiter contract template. "
            "threat type: sandwich sandwich attack buy target sell same block MEV. "
            "threat type: mev mev attack front running high frequency volume. "
            "threat type: novel_variant novel variant unknown attack generalisation. "
            "threat type: volume_attack high volume repeated transactions gradual drain. "
            "threat type: cross_protocol cross protocol manipulation. "
            "severity MEDIUM severity HIGH. "
            "recommended template: rate_limiter. "
            "use rate_limiter for sandwich MEDIUM. "
            "use rate_limiter for mev HIGH. "
            "use rate_limiter for novel variant unknown threat closest match. "
            "use rate_limiter for volume attack repeated high volume transactions. "
            "mechanism: throttles per block transaction volume cap configurable threshold. "
            "tracks cumulative volume per block rejects transactions exceeding limit. "
            "prevents volume exploits caps value flow within single block. "
            "deploy rate_limiter for repeated high volume not flash loan not wallet."
        ),
        "metadata": {
            "template_key":  "rate_limiter",
            "threat_types":  "volume_attack,sandwich,mev,high_frequency,novel_variant,gradual_drain",
            "severity":      "MEDIUM,HIGH",
            "mechanism":     "volume_cap",
            "description":   "Throttles per-block volume below configurable threshold",
            "openzeppelin":  "none",
            "deploy_time_s": "fast",
            "audited":       "true",
        },
        "solidity": textwrap.dedent("""\
            // SPDX-License-Identifier: MIT
            pragma solidity ^0.8.20;

            /**
             * RateLimiter_{{INCIDENT_ID}}.sol
             * ==================================
             * Pre-audited response template -- STAGE 1 RAG contract_templates
             *
             * PURPOSE:
             *   Autonomously deployed by ContractAgent to throttle per-block
             *   transaction volume when sandwich attacks, MEV exploits, or
             *   novel volume-based threats are detected.
             *
             * USE FOR:
             *   - Sandwich attacks (buy-target-sell in same block)
             *   - MEV / front-running high-volume patterns
             *   - Novel variant attacks (closest-match by RAG similarity)
             *   - Any MEDIUM anomaly with sustained high transaction volume
             *
             * MAPS TO FRAMEWORK:
             *   Section 4.1: "autonomous contract selection and parameterisation"
             *   RO5: deployed without human intervention
             *   Stage 5 novel_variant scenario: selected by RAG similarity
             *
             * AUDITED WITH: Mythril (offline, pre-deployment)
             * VERIFIED WITH: Slither (hot path, <2 seconds)
             */

            contract RateLimiter_{{INCIDENT_ID}} {

                address public immutable owner;
                uint256 public maxVolumePerBlock;  // in wei

                uint256 private _currentBlock;
                uint256 private _currentBlockVolume;

                event VolumeExceeded(
                    uint256 indexed blockNum,
                    uint256         volume,
                    uint256         limit
                );
                event LimitUpdated(
                    uint256 oldLimit,
                    uint256 newLimit,
                    uint256 timestamp
                );
                event VolumeRecorded(
                    uint256 indexed blockNum,
                    uint256         cumulativeVolume,
                    uint256         limit,
                    bool            allowed
                );

                constructor(uint256 _maxVolumeWei) {
                    require(_maxVolumeWei > 0, "Limit must be positive");
                    owner              = msg.sender;
                    maxVolumePerBlock  = _maxVolumeWei;
                }

                modifier onlyOwner() {
                    require(msg.sender == owner, "Not owner");
                    _;
                }

                /**
                 * Record transaction volume for the current block.
                 * Returns true if volume is within limit; false if exceeded.
                 * Resets volume counter on each new block automatically.
                 */
                function recordVolume(
                    uint256 amountWei
                ) external onlyOwner returns (bool allowed) {
                    // New block -- reset counter
                    if (block.number > _currentBlock) {
                        _currentBlock       = block.number;
                        _currentBlockVolume = 0;
                    }

                    _currentBlockVolume += amountWei;
                    allowed = (_currentBlockVolume <= maxVolumePerBlock);

                    if (!allowed) {
                        emit VolumeExceeded(
                            block.number,
                            _currentBlockVolume,
                            maxVolumePerBlock
                        );
                    }

                    emit VolumeRecorded(
                        block.number,
                        _currentBlockVolume,
                        maxVolumePerBlock,
                        allowed
                    );

                    return allowed;
                }

                function updateLimit(uint256 newLimitWei) external onlyOwner {
                    require(newLimitWei > 0, "Limit must be positive");
                    uint256 old = maxVolumePerBlock;
                    maxVolumePerBlock = newLimitWei;
                    emit LimitUpdated(old, newLimitWei, block.timestamp);
                }

                function getCurrentBlockVolume() external view returns (uint256) {
                    if (block.number > _currentBlock) return 0;
                    return _currentBlockVolume;
                }

                function isWithinLimit(uint256 additionalWei) external view returns (bool) {
                    uint256 vol = (block.number > _currentBlock)
                        ? additionalWei
                        : _currentBlockVolume + additionalWei;
                    return vol <= maxVolumePerBlock;
                }
            }
            """),
    },
}


class FraudKnowledgeAgent(BaseAgent):
    """
    RAG knowledge base with TWO collections:
      1. fraud_events        -- historical fraud detections
      2. contract_templates  -- pre-audited Solidity response templates

    Both wrap ChromaDB -- degrade gracefully if not installed.
    """

    FRAUD_COLLECTION     = "fraud_events"
    TEMPLATE_COLLECTION  = "contract_templates"

    def __init__(self, run_dir: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Parameters
        ----------
        run_dir : str
            Base run directory. ChromaDB will be stored at {run_dir}/rag_store/
        embedding_model : str
            Sentence-transformer model for embeddings.
            "all-MiniLM-L6-v2" is 80MB, fast, and free.
        """
        super().__init__(name="FraudKnowledgeAgent")
        self.run_dir         = run_dir
        self.store_dir       = os.path.join(run_dir, "rag_store")
        self.embedding_model = embedding_model

        # ChromaDB internals
        self._client              = None
        self._collection          = None   # fraud_events
        self._template_collection = None   # contract_templates
        self._ef                  = None
        self._available           = False

        os.makedirs(self.store_dir, exist_ok=True)
        self._init_store()

    # ════════════════════════════════════════════════════════
    # INITIALISATION
    # ════════════════════════════════════════════════════════

    def _init_store(self):
        """Initialise both ChromaDB collections. Silent on failure."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            self._client = chromadb.PersistentClient(path=self.store_dir)

            # ── Collection 1: fraud_events ────────────────────
            self._collection = self._client.get_or_create_collection(
                name=self.FRAUD_COLLECTION,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

            # ── Collection 2: contract_templates ─────────────
            self._template_collection = self._client.get_or_create_collection(
                name=self.TEMPLATE_COLLECTION,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

            self._available = True
            self.logger.info(
                f"[{self.name}] ChromaDB ready at {self.store_dir} | "
                f"fraud_events={self._collection.count()} docs | "
                f"contract_templates={self._template_collection.count()} docs"
            )

            # Auto-seed contract templates if collection is empty
            if self._template_collection.count() == 0:
                seeded = self.index_contract_templates()
                self.logger.info(
                    f"[{self.name}] Auto-seeded {seeded} contract templates "
                    f"into '{self.TEMPLATE_COLLECTION}' collection"
                )

        except ImportError:
            self.logger.warning(
                f"[{self.name}] chromadb / sentence-transformers not installed. "
                "RAG disabled -- pipeline runs unchanged. "
                "Install: pip install chromadb sentence-transformers"
            )
        except Exception as e:
            self.logger.warning(
                f"[{self.name}] ChromaDB init failed ({e}). RAG disabled."
            )

    # ════════════════════════════════════════════════════════
    # FRAUD EVENTS -- index + query
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _event_to_text(row: dict) -> str:
        """
        Convert a fraud_events.csv row into an embeddable text string.
        The text summarises key signals so the embedding captures
        semantically meaningful fraud patterns.
        """
        decision = row.get("decision", "UNKNOWN")
        action   = row.get("policy_action", "UNKNOWN")
        reason   = row.get("policy_reason", "UNKNOWN")
        risk     = float(row.get("risk_score", 0.0))
        p_rf     = float(row.get("p_rf", 0.0))
        s_if     = float(row.get("s_if", 0.0))
        wallet   = row.get("from_address", "unknown")
        batch    = row.get("batch", "?")
        y_true   = int(row.get("y_true", -1))
        label    = "confirmed fraud" if y_true == 1 else "false positive"

        return (
            f"Fraud event in batch {batch}. "
            f"Decision: {decision}. Policy action: {action}. Reason: {reason}. "
            f"Risk score: {risk:.3f}. RF probability: {p_rf:.3f}. "
            f"IF anomaly score: {s_if:.3f}. "
            f"Wallet: {wallet}. Ground truth: {label}."
        )

    def index_fraud_events(self, fraud_events_path: str) -> int:
        """
        Read fraud_events.csv and upsert all rows into the fraud_events collection.
        Upsert-safe: tx_hash is the document ID, so re-calling never duplicates.
        Returns number of documents indexed.
        """
        if not self._available:
            return 0
        if not os.path.exists(fraud_events_path):
            self.logger.info(
                f"[{self.name}] No fraud_events.csv yet -- nothing to index."
            )
            return 0

        try:
            docs, ids, metas = [], [], []
            with open(fraud_events_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc_id = str(row.get("tx_hash", "")).strip()
                    if not doc_id:
                        continue
                    text = self._event_to_text(row)
                    meta = {k: str(v) for k, v in row.items()}
                    docs.append(text)
                    ids.append(doc_id)
                    metas.append(meta)

            if not docs:
                return 0

            batch_size = 100
            total = 0
            for i in range(0, len(docs), batch_size):
                self._collection.upsert(
                    documents=docs[i:i + batch_size],
                    ids=ids[i:i + batch_size],
                    metadatas=metas[i:i + batch_size],
                )
                total += len(docs[i:i + batch_size])

            self.logger.info(
                f"[{self.name}] Indexed {total} fraud events | "
                f"total in store: {self._collection.count()}"
            )
            return total

        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Fraud event indexing failed ({e})"
            )
            return 0

    def query_similar(
        self,
        query_text: str,
        n_results: int = 3,
        confirmed_only: bool = True,
    ) -> list[dict]:
        """
        Find the top-k most similar historical fraud events.

        Parameters
        ----------
        query_text : str
            Text describing the current transaction/threat.
        n_results : int
            Number of similar events to retrieve.
        confirmed_only : bool
            If True, only return events where y_true == "1".

        Returns
        -------
        list of dicts with keys: text, metadata, distance
        """
        if not self._available or self._collection.count() == 0:
            return []

        try:
            where   = {"y_true": "1"} if confirmed_only else None
            count   = self._collection.count()
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(n_results, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "text":     doc,
                    "metadata": meta,
                    "distance": float(dist),
                })
            return output

        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Fraud events query failed ({e})"
            )
            return []

    def build_rag_context(
        self,
        risk_score: float,
        p_rf: float,
        s_if: float,
        decision: str,
        wallet: str,
        n_results: int = 3,
    ) -> str:
        """
        Build a human-readable RAG context string for audit logs and LLM prompts.
        Returns empty string if RAG unavailable.
        """
        query_text = (
            f"risk_score={risk_score:.3f} p_rf={p_rf:.3f} s_if={s_if:.3f} "
            f"decision={decision} wallet={wallet}"
        )
        similar = self.query_similar(query_text, n_results=n_results)
        if not similar:
            return ""

        lines = [f"RAG context -- {len(similar)} similar past fraud events retrieved:"]
        for i, item in enumerate(similar, 1):
            m = item["metadata"]
            lines.append(
                f"  [{i}] batch={m.get('batch','?')} "
                f"decision={m.get('decision','?')} "
                f"action={m.get('policy_action','?')} "
                f"risk={float(m.get('risk_score',0)):.3f} "
                f"similarity={1 - item['distance']:.3f} "
                f"label={'fraud' if m.get('y_true') == '1' else 'fp'}"
            )
        return "\n".join(lines)

    # ════════════════════════════════════════════════════════
    # CONTRACT TEMPLATES -- index + query
    # ════════════════════════════════════════════════════════

    def index_contract_templates(
        self,
        templates: dict = None,
    ) -> int:
        """
        Index all pre-audited Solidity templates into the contract_templates collection.

        Called automatically at init if the collection is empty.
        Can be called again to re-index after adding new templates.

        Parameters
        ----------
        templates : dict, optional
            Template dict in CONTRACT_TEMPLATES format.
            Defaults to the built-in CONTRACT_TEMPLATES at module level.

        Returns
        -------
        int -- number of templates indexed.
        """
        if not self._available:
            return 0

        if templates is None:
            templates = CONTRACT_TEMPLATES

        try:
            docs, ids, metas = [], [], []
            for key, tmpl in templates.items():
                doc_id = f"template_{key}"
                text   = tmpl["text"]
                meta   = {
                    **tmpl["metadata"],
                    "indexed_at": datetime.utcnow().isoformat(),
                    "doc_id":     doc_id,
                }
                # Store Solidity source in metadata (truncated for ChromaDB limits)
                sol = tmpl.get("solidity", "")
                meta["solidity_preview"] = sol[:500]  # first 500 chars as preview

                docs.append(text)
                ids.append(doc_id)
                metas.append(meta)

            if not docs:
                return 0

            self._template_collection.upsert(
                documents=docs,
                ids=ids,
                metadatas=metas,
            )

            self.logger.info(
                f"[{self.name}] Indexed {len(docs)} contract templates | "
                f"total: {self._template_collection.count()}"
            )
            return len(docs)

        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Template indexing failed ({e})"
            )
            return 0

    def query_template(
        self,
        query_text: str,
        n_results: int = 1,
    ) -> list[dict]:
        """
        Find the best-matching contract template for a given threat description.

        Called by ContractAgent._select_template() to replace the built-in
        dict fallback with a real RAG-based similarity search.

        Parameters
        ----------
        query_text : str
            Description of the threat, e.g.:
            "contract template for flash_loan severity=CRITICAL"
        n_results : int
            Number of results to return (default 1 = best match only).

        Returns
        -------
        list of dicts, each with keys:
            text      : the stored document text
            metadata  : template metadata including template_key
            distance  : cosine distance (lower = more similar)
            similarity: 1 - distance (higher = more similar)

        Example result:
            [{"text": "Contract template: circuit_breaker ...",
              "metadata": {"template_key": "circuit_breaker", ...},
              "distance": 0.08,
              "similarity": 0.92}]
        """
        if not self._available or self._template_collection.count() == 0:
            return []

        try:
            count = self._template_collection.count()
            results = self._template_collection.query(
                query_texts=[query_text],
                n_results=min(n_results, count),
                include=["documents", "metadatas", "distances"],
            )

            output = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "text":       doc,
                    "metadata":   meta,
                    "distance":   float(dist),
                    "similarity": round(1.0 - float(dist), 4),
                })
            return output

        except Exception as e:
            self.logger.warning(
                f"[{self.name}] Template query failed ({e})"
            )
            return []

    def get_template_solidity(self, template_key: str) -> str:
        """
        Return the full Solidity source for a given template key.
        Looks up CONTRACT_TEMPLATES at module level (not ChromaDB,
        since ChromaDB truncates large strings in metadata).

        Returns empty string if key not found.
        """
        tmpl = CONTRACT_TEMPLATES.get(template_key)
        if tmpl:
            return tmpl.get("solidity", "")
        return ""

    def list_templates(self) -> list[str]:
        """Return list of indexed template keys."""
        return list(CONTRACT_TEMPLATES.keys())

    # ════════════════════════════════════════════════════════
    # STORE SIZE UTILITIES
    # ════════════════════════════════════════════════════════

    def get_store_size(self) -> int:
        """Return total documents in the fraud_events collection."""
        if not self._available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def get_template_store_size(self) -> int:
        """Return total documents in the contract_templates collection."""
        if not self._available:
            return 0
        try:
            return self._template_collection.count()
        except Exception:
            return 0

    def get_all_store_sizes(self) -> dict:
        """Return sizes of both collections."""
        return {
            "fraud_events":       self.get_store_size(),
            "contract_templates": self.get_template_store_size(),
        }

    # ════════════════════════════════════════════════════════
    # BaseAgent._run() -- satisfies interface
    # ════════════════════════════════════════════════════════

    def _run(self, msg: AgentMessage) -> AgentMessage:
        """
        Batch-mode interface: indexes fraud_events.csv from run_dir.
        Called by CoordinatorAgent after each batch.
        """
        run_dir           = msg.payload.get("run_dir", self.run_dir)
        fraud_events_path = os.path.join(run_dir, "fraud_events.csv")
        indexed           = self.index_fraud_events(fraud_events_path)

        return AgentMessage(
            sender=self.name,
            payload={
                "rag_indexed":            indexed,
                "rag_store_size":         self.get_store_size(),
                "rag_template_store_size": self.get_template_store_size(),
                **{k: v for k, v in msg.payload.items()},
            },
            status="ok",
        )