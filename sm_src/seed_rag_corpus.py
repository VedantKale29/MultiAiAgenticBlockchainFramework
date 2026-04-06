"""
seed_rag_corpus.py
===================
STAGE 1 EXTENSION — External Threat Intelligence Seeding

WHAT THIS SCRIPT DOES:
  Seeds the RAG knowledge base (ChromaDB fraud_events collection) with two
  categories of external threat intelligence BEFORE the main pipeline runs.
  This makes DecisionAgent's RAG retrieval richer and more accurate from
  batch 1 — instead of starting cold.

WHY IT EXISTS:
  The framework doc (Stage 1, Section 9.2) specifies:
    "Fetch and index CVE/NVD entries for Ethereum vulnerabilities via NVD API"
    "Index Rekt.news flash loan incident post-mortems as RAG corpus entries"

  Without this seeding, RAG only knows about fraud_events.csv detections.
  After seeding, it also knows:
    - Real CVE records for Solidity/EVM vulnerabilities (from NVD REST API)
    - 9 real DeFi incident post-mortems (Ronin, Nomad, Beanstalk, etc.)
  This is what turns RAG from a replay log into a genuine threat knowledge base.

TWO SEEDING SOURCES:

  Source 1 — NVD/CVE API (live fetch, free, no key required):
    URL: https://services.nvd.nist.gov/rest/json/cves/2.0
    Searches: "ethereum", "solidity", "defi", "smart contract", "blockchain"
    Extracts: CVE ID, description, CVSS severity, published date
    Indexed as: fraud_events collection documents with source="nvd_cve"

  Source 2 — DeFi Incident Post-Mortems (hardcoded, always available):
    9 real incidents with: name, date, loss_usd, attack_type, mechanism,
    attacker_actions, root_cause, recommended_template
    Sources: Rekt.news, official post-mortems, public blockchain analysis
    Indexed as: fraud_events collection documents with source="defi_incident"

RETRIEVAL QUALITY CHECK:
  After seeding, runs 3 test queries and prints cosine similarity scores
  so you can verify the knowledge base is working before running the pipeline.

USAGE:
  # Run ONCE before run_pipeline.py:
  python seed_rag_corpus.py

  # Re-run safely at any time (upsert-safe, no duplicates):
  python seed_rag_corpus.py

  # Specify a run_dir (must match run_pipeline.py's output dir):
  python seed_rag_corpus.py --run-dir runs/run_42

  # Skip NVD fetch (use hardcoded incidents only):
  python seed_rag_corpus.py --no-nvd

  # Verbose: show each document being indexed:
  python seed_rag_corpus.py --verbose

OUTPUT:
  - Seeds fraud_events ChromaDB collection in {run_dir}/rag_store/
  - Prints before/after store size
  - Prints retrieval quality check (3 test queries with similarity scores)
  - Saves seed_report.json to run_dir

INSTALL:
  pip install chromadb sentence-transformers requests
"""

import os
import sys
import json
import time
import logging
import argparse
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SeedRAG")

SEP  = "=" * 68
SEP2 = "-" * 68


# ══════════════════════════════════════════════════════════════════════════
# HARDCODED DEFI INCIDENT POST-MORTEMS
# Sources: Rekt.news, official post-mortems, public blockchain analysis
# Format mirrors fraud_events.csv schema so DecisionAgent reads them natively
# ══════════════════════════════════════════════════════════════════════════

DEFI_INCIDENTS = [
    {
        "incident_id":    "ronin_bridge_2022",
        "name":           "Ronin Bridge Hack",
        "date":           "2022-03-29",
        "loss_usd":       625_000_000,
        "attack_type":    "bridge_exploit",
        "chain":          "Ethereum / Ronin",
        "mechanism": (
            "Attacker compromised 5 of 9 Ronin validator private keys "
            "via spear-phishing and a backdoored PDF. Used compromised keys "
            "to forge withdrawal signatures. Drained 173,600 ETH and 25.5M USDC "
            "across two transactions. Bridge had reduced validator threshold "
            "from 5/9 to 4/9 weeks earlier — not reverted after an event."
        ),
        "attacker_actions": (
            "private key compromise, signature forgery, bridge drain, "
            "large ETH transfer, USDC drain, two-transaction drain"
        ),
        "root_cause": (
            "Insufficient validator key security, reduced multisig threshold, "
            "no anomaly detection on large withdrawals, no rate limiting on bridge"
        ),
        "recommended_template": "circuit_breaker",
        "threat_keywords": (
            "bridge exploit validator compromise private key phishing "
            "large withdrawal signature forgery multisig threshold "
            "ETH drain USDC drain cross-chain bridge attack"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/ronin-rekt/",
    },
    {
        "incident_id":    "nomad_bridge_2022",
        "name":           "Nomad Bridge Hack",
        "date":           "2022-08-01",
        "loss_usd":       190_000_000,
        "attack_type":    "bridge_exploit",
        "chain":          "Ethereum",
        "mechanism": (
            "A routine upgrade introduced a bug: the zero hash 0x00 was "
            "accepted as a valid Merkle root, meaning ANY message could be "
            "proved valid. Exploit was copycat — hundreds of addresses "
            "replayed the initial attack transaction by changing the recipient "
            "address. Bridge drained in under 2 hours."
        ),
        "attacker_actions": (
            "merkle root bypass, message replay, copycat drain, "
            "hundreds of wallets, 2-hour mass drain, upgrade bug exploitation"
        ),
        "root_cause": (
            "Smart contract upgrade bug in Replica contract, "
            "zero-value Merkle root accepted as valid proof, "
            "no invariant checks on upgrade, no circuit breaker for mass drain"
        ),
        "recommended_template": "circuit_breaker",
        "threat_keywords": (
            "bridge exploit merkle root bypass message replay upgrade bug "
            "mass drain copycat attack hundreds wallets 2 hours ETH drain "
            "contract upgrade vulnerability proof bypass"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/nomad-rekt/",
    },
    {
        "incident_id":    "beanstalk_2022",
        "name":           "Beanstalk Farms Flash Loan Governance Attack",
        "date":           "2022-04-17",
        "loss_usd":       182_000_000,
        "attack_type":    "flash_loan_governance",
        "chain":          "Ethereum",
        "mechanism": (
            "Attacker used a flash loan to acquire a supermajority of "
            "governance tokens (BEAN) within a single transaction block, "
            "then immediately passed a malicious governance proposal that "
            "transferred all protocol funds to the attacker. "
            "Flash loan repaid in the same transaction."
        ),
        "attacker_actions": (
            "flash loan governance attack, token acquisition same block, "
            "malicious proposal immediate pass, protocol fund drain, "
            "flash loan repayment same transaction, 13 second attack"
        ),
        "root_cause": (
            "No time-lock on governance execution, flash loan attack vector "
            "on voting power, immediate proposal execution without delay, "
            "no flash loan protection in governance contract"
        ),
        "recommended_template": "circuit_breaker",
        "threat_keywords": (
            "flash loan governance attack governance token acquisition "
            "supermajority single block malicious proposal protocol drain "
            "no timelock flash loan protection voting power manipulation "
            "rapid governance exploit DeFi governance attack"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/beanstalk-rekt/",
    },
    {
        "incident_id":    "cream_finance_2021",
        "name":           "Cream Finance Flash Loan Reentrancy",
        "date":           "2021-10-27",
        "loss_usd":       130_000_000,
        "attack_type":    "flash_loan_reentrancy",
        "chain":          "Ethereum",
        "mechanism": (
            "Two-step flash loan attack. Attacker borrowed 2 billion CREAM "
            "using a flash loan, then exploited a reentrancy vulnerability in "
            "the lending contract to manipulate price oracle and borrow against "
            "inflated collateral. Price oracle read from an AMM pool that "
            "the flash loan had temporarily distorted."
        ),
        "attacker_actions": (
            "flash loan reentrancy oracle manipulation collateral inflation "
            "borrow against inflated collateral multi-step flash loan "
            "AMM price manipulation lending protocol drain"
        ),
        "root_cause": (
            "Reentrancy vulnerability in lending contract, "
            "price oracle reading from manipulable AMM pool, "
            "no flash loan guard on borrow function, "
            "no price deviation check before accepting collateral"
        ),
        "recommended_template": "circuit_breaker",
        "threat_keywords": (
            "flash loan reentrancy price oracle manipulation lending protocol "
            "collateral inflation borrow exploit AMM price manipulation "
            "reentrancy attack flash loan guard oracle deviation "
            "DeFi lending attack flash loan reentrancy"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/cream-rekt-2/",
    },
    {
        "incident_id":    "wormhole_2022",
        "name":           "Wormhole Bridge Signature Verification Bypass",
        "date":           "2022-02-02",
        "loss_usd":       320_000_000,
        "attack_type":    "bridge_exploit",
        "chain":          "Solana / Ethereum",
        "mechanism": (
            "Attacker exploited a deprecated function in the Wormhole bridge "
            "that bypassed guardian signature verification. The function "
            "verify_signatures was incorrectly marked as a system call instead "
            "of a user-space call, allowing the attacker to spoof 120,000 wETH "
            "mint authorization without valid guardian signatures."
        ),
        "attacker_actions": (
            "signature verification bypass deprecated function exploit "
            "unauthorized mint 120000 wETH bridge exploit guardian bypass "
            "cross-chain exploit"
        ),
        "root_cause": (
            "Deprecated function left in production, incorrect syscall designation, "
            "no guardian signature validation on mint, "
            "missing access control on critical bridge function"
        ),
        "recommended_template": "address_blocklist",
        "threat_keywords": (
            "bridge exploit signature bypass deprecated function unauthorized mint "
            "guardian bypass cross-chain attack wETH mint spoof "
            "access control missing bridge signature verification bypass"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/wormhole-rekt/",
    },
    {
        "incident_id":    "euler_finance_2023",
        "name":           "Euler Finance Flash Loan Donation Attack",
        "date":           "2023-03-13",
        "loss_usd":       197_000_000,
        "attack_type":    "flash_loan",
        "chain":          "Ethereum",
        "mechanism": (
            "Attacker exploited a missing health-check in the donateToReserves "
            "function. By taking a flash loan, minting eTokens (collateral), "
            "donating a large amount to reserves (which reduced their own "
            "collateral value), and triggering liquidation of their own position, "
            "the attacker extracted more value than deposited. "
            "Attack required 9 nested contract calls."
        ),
        "attacker_actions": (
            "flash loan donation attack health check bypass eToken mint "
            "donate to reserves collateral reduction self-liquidation "
            "9 nested contract calls multi-step flash loan"
        ),
        "root_cause": (
            "Missing health check in donateToReserves function, "
            "self-liquidation not prevented, "
            "collateral accounting error after donation, "
            "complex nested call allowed without invariant check"
        ),
        "recommended_template": "circuit_breaker",
        "threat_keywords": (
            "flash loan donation attack health check missing self-liquidation "
            "eToken collateral manipulation nested calls DeFi lending exploit "
            "flash loan complex multi-step attack euler finance"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/euler-rekt/",
    },
    {
        "incident_id":    "mango_markets_2022",
        "name":           "Mango Markets Oracle Manipulation",
        "date":           "2022-10-11",
        "loss_usd":       117_000_000,
        "attack_type":    "oracle_manipulation",
        "chain":          "Solana",
        "mechanism": (
            "Attacker opened large long position in MNGO perpetuals, "
            "then purchased large amounts of spot MNGO to artificially "
            "inflate the oracle price by 10x. Used inflated collateral "
            "value from manipulated price to borrow all available liquidity "
            "across supported tokens from the Mango protocol treasury."
        ),
        "attacker_actions": (
            "oracle price manipulation spot purchase collateral inflation "
            "borrow against inflated oracle drain treasury "
            "perpetuals position manipulation 10x price pump"
        ),
        "root_cause": (
            "Oracle price taken directly from thin spot market, "
            "no TWAP or circuit breaker on price deviation, "
            "no borrow cap relative to oracle confidence interval, "
            "no rate limiting on treasury borrows"
        ),
        "recommended_template": "rate_limiter",
        "threat_keywords": (
            "oracle manipulation price manipulation oracle attack "
            "spot market thin liquidity collateral inflation borrow drain "
            "TWAP missing price deviation no circuit breaker "
            "treasury drain oracle price manipulation DeFi"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/mango-markets-rekt/",
    },
    {
        "incident_id":    "bzx_fulcrum_2020",
        "name":           "bZx Fulcrum Flash Loan Oracle Attack",
        "date":           "2020-02-15",
        "loss_usd":       954_000,
        "attack_type":    "flash_loan",
        "chain":          "Ethereum",
        "mechanism": (
            "First major flash loan attack. Attacker borrowed 10,000 ETH "
            "from dYdX, used 5,500 ETH as collateral on Compound to borrow "
            "112 WBTC, then sold WBTC on Uniswap to crash the ETH/BTC price "
            "on the oracle that bZx used. Simultaneously shorted ETH on bZx "
            "Fulcrum, profiting from the oracle-manipulated price. "
            "All in one transaction block."
        ),
        "attacker_actions": (
            "flash loan oracle manipulation first flash loan attack "
            "cross-protocol attack dYdX Compound Uniswap bZx "
            "single transaction block WBTC price crash short position profit"
        ),
        "root_cause": (
            "Oracle relied on single DEX spot price, "
            "no flash loan protection on borrow, "
            "cross-protocol composability risk, "
            "price easily manipulated in single transaction"
        ),
        "recommended_template": "circuit_breaker",
        "threat_keywords": (
            "flash loan oracle manipulation cross-protocol attack single block "
            "DEX price manipulation first flash loan attack bZx oracle attack "
            "composability risk single transaction profit ETH price manipulation"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/bzx-rekt/",
    },
    {
        "incident_id":    "poly_network_2021",
        "name":           "Poly Network Cross-Chain Access Control Bypass",
        "date":           "2021-08-10",
        "loss_usd":       611_000_000,
        "attack_type":    "access_control",
        "chain":          "Ethereum / BSC / Polygon",
        "mechanism": (
            "Attacker found a cross-chain message relay function that allowed "
            "any caller to override the keeper (trusted relayer) address. "
            "By sending a crafted cross-chain message, the attacker replaced "
            "the EthCrossChainManager's keeper with their own address, "
            "then used that control to drain funds across three chains. "
            "Largest DeFi hack at the time."
        ),
        "attacker_actions": (
            "access control bypass keeper replacement cross-chain message "
            "crafted payload privilege escalation multi-chain drain "
            "trusted relayer override contract takeover"
        ),
        "root_cause": (
            "Cross-chain message handler allowed keeper replacement by anyone, "
            "missing access control on critical setter function, "
            "no validation of cross-chain message origin, "
            "no multi-sig on keeper address changes"
        ),
        "recommended_template": "address_blocklist",
        "threat_keywords": (
            "access control bypass cross-chain exploit keeper replacement "
            "privilege escalation crafted message contract takeover "
            "multi-chain drain trusted relayer override missing access control "
            "cross-chain bridge access control vulnerability"
        ),
        "source": "rekt.news",
        "source_url": "https://rekt.news/polynetwork-rekt/",
    },
]


# ══════════════════════════════════════════════════════════════════════════
# NVD CVE FETCHER
# ══════════════════════════════════════════════════════════════════════════

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_KEYWORDS = ["ethereum", "solidity", "defi", "smart contract reentrancy", "erc20"]
NVD_MAX_PER_KEYWORD = 10   # keep it small — free tier has rate limits


def fetch_nvd_cves(keywords: list, max_per_keyword: int, verbose: bool = False) -> list:
    """
    Fetch CVE records from the NVD REST API (no API key required).
    Returns a list of dicts ready for ChromaDB indexing.

    Rate limit: NVD allows ~5 requests/30s without a key.
    We sleep 6s between keyword requests to stay well within limits.
    """
    records = []
    seen_ids = set()

    session = requests.Session()
    session.headers.update({"User-Agent": "AgenAI-BlockchainFraud-RAGSeeder/1.0"})

    for keyword in keywords:
        log.info(f"  NVD: fetching CVEs for '{keyword}'...")
        try:
            resp = session.get(
                NVD_BASE_URL,
                params={
                    "keywordSearch": keyword,
                    "resultsPerPage": max_per_keyword,
                    "startIndex": 0,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            log.warning(f"  NVD: no internet connection — skipping '{keyword}'")
            continue
        except requests.exceptions.Timeout:
            log.warning(f"  NVD: request timed out for '{keyword}'")
            continue
        except requests.exceptions.HTTPError as e:
            log.warning(f"  NVD: HTTP error for '{keyword}': {e}")
            continue
        except Exception as e:
            log.warning(f"  NVD: unexpected error for '{keyword}': {e}")
            continue

        vulnerabilities = data.get("vulnerabilities", [])
        fetched = 0
        for item in vulnerabilities:
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            if not cve_id or cve_id in seen_ids:
                continue
            seen_ids.add(cve_id)

            # Extract description (prefer English)
            descriptions = cve.get("descriptions", [])
            description  = ""
            for d in descriptions:
                if d.get("lang") == "en":
                    description = d.get("value", "")
                    break
            if not description:
                continue

            # Extract CVSS severity
            severity   = "UNKNOWN"
            cvss_score = 0.0
            metrics    = cve.get("metrics", {})
            for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if metric_key in metrics and metrics[metric_key]:
                    m = metrics[metric_key][0]
                    severity   = m.get("cvssData", {}).get("baseSeverity", "UNKNOWN")
                    cvss_score = m.get("cvssData", {}).get("baseScore", 0.0)
                    break

            published = cve.get("published", "")[:10]

            record = {
                "incident_id":      cve_id.lower().replace("-", "_"),
                "name":             cve_id,
                "date":             published,
                "loss_usd":         0,
                "attack_type":      "cve_vulnerability",
                "chain":            "ethereum",
                "mechanism":        description,
                "attacker_actions": f"vulnerability: {description[:200]}",
                "root_cause":       description,
                "recommended_template": _cve_to_template(description),
                "threat_keywords":  f"CVE vulnerability {keyword} {severity} {description[:100]}",
                "source":           "nvd_cve",
                "source_url":       f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                "cvss_score":       cvss_score,
                "cvss_severity":    severity,
            }
            records.append(record)
            fetched += 1
            if verbose:
                log.info(f"    {cve_id}: {severity} ({cvss_score}) — {description[:80]}...")

        log.info(f"  NVD: {fetched} new CVEs for '{keyword}'")

        # Respect NVD rate limit: 5 requests per 30s without API key
        time.sleep(6)

    return records


def _cve_to_template(description: str) -> str:
    """Heuristically map CVE description to the best response template."""
    desc = description.lower()
    if any(w in desc for w in ["reentranc", "reentry", "recursive call", "callback"]):
        return "circuit_breaker"
    if any(w in desc for w in ["flash loan", "flashloan", "single block", "price oracle"]):
        return "circuit_breaker"
    if any(w in desc for w in ["access control", "privilege", "unauthorized", "bypass", "spoofing"]):
        return "address_blocklist"
    if any(w in desc for w in ["overflow", "underflow", "integer", "arithmetic"]):
        return "address_blocklist"
    if any(w in desc for w in ["dos", "denial", "rate", "throttle", "spam"]):
        return "rate_limiter"
    return "address_blocklist"   # conservative default


# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT FORMATTER
# ══════════════════════════════════════════════════════════════════════════

def _incident_to_chromadb_doc(record: dict) -> tuple[str, str, dict]:
    """
    Convert an incident/CVE record into a ChromaDB (text, id, metadata) triple.

    The text field is what gets embedded. Written to be semantically rich:
    - Repeats key threat terms in multiple phrasings
    - Includes attack_type, mechanism, keywords, recommended_template
    This ensures cosine similarity queries return high scores for short queries
    like "flash loan attack" or "reentrancy CRITICAL".
    """
    source   = record.get("source", "unknown")
    inc_id   = record.get("incident_id", "")
    name     = record.get("name", "")
    atype    = record.get("attack_type", "unknown")
    mech     = record.get("mechanism", "")
    actions  = record.get("attacker_actions", "")
    cause    = record.get("root_cause", "")
    keywords = record.get("threat_keywords", "")
    template = record.get("recommended_template", "address_blocklist")
    loss     = record.get("loss_usd", 0)
    date     = record.get("date", "")
    chain    = record.get("chain", "ethereum")

    # Rich embedding text — keyword-dense for all-MiniLM-L6-v2
    text = (
        f"incident: {name}. "
        f"attack type: {atype}. "
        f"threat type: {atype}. "
        f"chain: {chain}. "
        f"date: {date}. "
        f"mechanism: {mech} "
        f"attacker actions: {actions}. "
        f"root cause: {cause}. "
        f"threat keywords: {keywords}. "
        f"recommended response template: {template}. "
        f"use {template} for {atype} attack. "
        f"loss: {'${:,.0f}'.format(loss) if loss else 'CVE vulnerability'}. "
        f"source: {source}."
    )

    # Deterministic doc ID from incident_id + source
    doc_id = f"{source}_{inc_id}"

    # Metadata — mirrors fraud_events.csv schema where possible
    metadata = {
        "incident_id":           inc_id,
        "name":                  name,
        "attack_type":           atype,
        "chain":                 chain,
        "date":                  date,
        "loss_usd":              str(loss),
        "recommended_template":  template,
        "source":                source,
        "source_url":            record.get("source_url", ""),
        "cvss_severity":         record.get("cvss_severity", ""),
        "cvss_score":            str(record.get("cvss_score", "")),
        # Align with fraud_events.csv fields so query_similar() works
        "y_true":                "1",          # confirmed threat
        "decision":              "AUTO-BLOCK",
        "risk_score":            str(record.get("cvss_score", "0.95") or "0.95"),
    }

    return text, doc_id, metadata


# ══════════════════════════════════════════════════════════════════════════
# CHROMADB UPSERT
# ══════════════════════════════════════════════════════════════════════════

def upsert_to_collection(collection, records: list, batch_size: int = 50,
                         verbose: bool = False) -> int:
    """Upsert a list of incident records into a ChromaDB collection."""
    docs, ids, metas = [], [], []
    for record in records:
        text, doc_id, metadata = _incident_to_chromadb_doc(record)
        docs.append(text)
        ids.append(doc_id)
        metas.append(metadata)
        if verbose:
            log.info(f"    Preparing: {doc_id} ({metadata.get('attack_type','?')})")

    if not docs:
        return 0

    total = 0
    for i in range(0, len(docs), batch_size):
        collection.upsert(
            documents=docs[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metas[i:i + batch_size],
        )
        total += len(docs[i:i + batch_size])

    return total


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVAL QUALITY CHECK
# ══════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    {
        "query":    "flash loan attack rapid drain single block ETH",
        "expected": "circuit_breaker",
        "label":    "Flash loan → circuit_breaker",
    },
    {
        "query":    "reentrancy attack repeated external call recursive vulnerability",
        "expected": "circuit_breaker",
        "label":    "Reentrancy → circuit_breaker",
    },
    {
        "query":    "oracle price manipulation collateral inflation borrow exploit",
        "expected": "rate_limiter",
        "label":    "Oracle manipulation → rate_limiter",
    },
    {
        "query":    "access control bypass privilege escalation address blocklist",
        "expected": "address_blocklist",
        "label":    "Access control → address_blocklist",
    },
]


def run_retrieval_check(collection, n_results: int = 3) -> list:
    """
    Run test queries against the seeded collection.
    Returns list of check results for the report.
    """
    results = []
    log.info("")
    log.info("  Retrieval quality check:")
    log.info(f"  {'Query label':<40} {'Top match':<25} {'Similarity':>10}")
    log.info("  " + "-" * 78)

    for q in TEST_QUERIES:
        try:
            count = collection.count()
            if count == 0:
                log.warning("  Collection empty — skipping retrieval check")
                break

            qr = collection.query(
                query_texts=[q["query"]],
                n_results=min(n_results, count),
                include=["documents", "metadatas", "distances"],
            )

            top_meta     = qr["metadatas"][0][0] if qr["metadatas"][0] else {}
            top_distance = qr["distances"][0][0]  if qr["distances"][0]  else 1.0
            similarity   = max(0.0, 1.0 - top_distance)   # cosine: dist → similarity

            top_name     = top_meta.get("name", top_meta.get("incident_id", "?"))[:24]
            top_template = top_meta.get("recommended_template", "?")

            match_icon = "[OK]" if top_template == q["expected"] else "[??]"

            log.info(
                f"  {match_icon} {q['label']:<40} "
                f"{top_name:<25} "
                f"{similarity:>10.3f}"
            )

            results.append({
                "query":            q["label"],
                "top_match_name":   top_name,
                "top_template":     top_template,
                "expected_template": q["expected"],
                "similarity":       round(similarity, 4),
                "passed":           top_template == q["expected"],
            })

        except Exception as e:
            log.warning(f"  Query failed: {q['label']} — {e}")
            results.append({"query": q["label"], "error": str(e), "passed": False})

    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Seed the RAG knowledge base with CVE/NVD + DeFi incident data."
    )
    p.add_argument("--run-dir", type=str, default=None,
                   help="RAG store directory (default: runs/run_42)")
    p.add_argument("--no-nvd",  action="store_true",
                   help="Skip NVD API fetch (use hardcoded incidents only)")
    p.add_argument("--verbose", action="store_true",
                   help="Print each document being indexed")
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.perf_counter()

    print()
    print(SEP)
    print("  RAG CORPUS SEEDER")
    print("  Stage 1 Extension — External Threat Intelligence")
    print(SEP)
    print()

    # ── Resolve run_dir ──────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Default to run_42 (paper seed)
        run_dir = ROOT / "runs" / "run_42"
        if not run_dir.exists():
            # Try to find any existing run dir
            runs_root = ROOT / "runs"
            if runs_root.exists():
                candidates = sorted(runs_root.iterdir(), reverse=True)
                if candidates:
                    run_dir = candidates[0]
                    log.info(f"Using most recent run dir: {run_dir}")
            else:
                run_dir = ROOT / "runs" / "run_42"

    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"RAG store location: {run_dir / 'rag_store'}")

    # ── Import ChromaDB via FraudKnowledgeAgent ──────────────────
    # We use FraudKnowledgeAgent to get the exact same ChromaDB client
    # and collection names that the pipeline uses — no duplication of config.
    try:
        from agents.fraud_knowledge_agent import FraudKnowledgeAgent
        agent = FraudKnowledgeAgent(run_dir=str(run_dir))
    except ImportError as e:
        log.error(f"Cannot import FraudKnowledgeAgent: {e}")
        log.error("Make sure you're running from sm_src/ with venv activated.")
        sys.exit(1)

    if not agent._available:
        log.error("ChromaDB not available.")
        log.error("Install: pip install chromadb sentence-transformers")
        sys.exit(1)

    collection = agent._collection     # fraud_events collection
    size_before = collection.count()
    log.info(f"fraud_events collection: {size_before} documents before seeding")

    # ── Source 1: DeFi Incident Post-Mortems ─────────────────────
    print()
    log.info(SEP2)
    log.info("SOURCE 1 — DeFi Incident Post-Mortems (hardcoded)")
    log.info(SEP2)
    n_incidents = upsert_to_collection(
        collection, DEFI_INCIDENTS, verbose=args.verbose
    )
    log.info(f"  Upserted {n_incidents} incident records")

    # ── Source 2: NVD CVE API ─────────────────────────────────────
    cve_records = []
    if not args.no_nvd:
        print()
        log.info(SEP2)
        log.info("SOURCE 2 — NVD/CVE API (live fetch)")
        log.info(SEP2)
        log.info(f"  Keywords: {NVD_KEYWORDS}")
        log.info(f"  Max per keyword: {NVD_MAX_PER_KEYWORD}")
        log.info("  (6s sleep between requests to respect NVD rate limit)")
        print()
        cve_records = fetch_nvd_cves(
            NVD_KEYWORDS, NVD_MAX_PER_KEYWORD, verbose=args.verbose
        )
        if cve_records:
            n_cves = upsert_to_collection(
                collection, cve_records, verbose=args.verbose
            )
            log.info(f"  Upserted {n_cves} CVE records")
        else:
            log.info("  No CVE records fetched (offline or API unavailable)")
    else:
        log.info("NVD fetch skipped (--no-nvd)")

    # ── Summary ───────────────────────────────────────────────────
    size_after = collection.count()
    print()
    log.info(SEP)
    log.info("SEEDING COMPLETE")
    log.info(SEP)
    log.info(f"  Before : {size_before} documents")
    log.info(f"  After  : {size_after} documents")
    log.info(f"  Added  : {size_after - size_before} new documents")
    log.info(f"    DeFi incidents : {n_incidents}")
    log.info(f"    CVE records    : {len(cve_records)}")

    # ── Retrieval quality check ───────────────────────────────────
    print()
    log.info(SEP2)
    log.info("RETRIEVAL QUALITY CHECK")
    log.info(SEP2)
    check_results = run_retrieval_check(collection)

    passed = sum(1 for r in check_results if r.get("passed"))
    total  = len(check_results)
    print()
    log.info(f"  {passed}/{total} queries returned expected template")

    if passed == total:
        log.info("  [PASS] RAG retrieval working correctly")
    elif passed >= total // 2:
        log.info("  [WARN] Some queries off — may improve with more data")
    else:
        log.warning("  [FAIL] Low retrieval quality — check embedding model")

    # ── Save report ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    report = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "run_dir":         str(run_dir),
        "size_before":     size_before,
        "size_after":      size_after,
        "added":           size_after - size_before,
        "incidents_added": n_incidents,
        "cves_added":      len(cve_records),
        "retrieval_checks": check_results,
        "elapsed_s":       round(elapsed, 2),
    }
    report_path = run_dir / "seed_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"  Report saved: {report_path}")

    print()
    print(SEP)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  RAG store ready at: {run_dir / 'rag_store'}")
    print(f"  Run next: python run_pipeline.py")
    print(SEP)
    print()


if __name__ == "__main__":
    main()