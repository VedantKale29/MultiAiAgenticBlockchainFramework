"""
mcp_server.py
=============
STAGE 1+2 -- MCP Server (standalone, does NOT modify existing pipeline)

WHAT THIS IS:
  A Model Context Protocol server that exposes your fraud detection
  system as callable tools. Any MCP-compatible client (Claude Desktop,
  LangGraph, custom scripts) can call these tools without touching
  your existing main.py or any agent.

HOW IT WORKS:
  This file is a SEPARATE process. You run it alongside your existing
  system, not instead of it:

      # Terminal 1 -- run pipeline as normal (unchanged)
      python main.py

      # Terminal 2 -- MCP server (new, optional)
      python mcp_server.py --run_dir runs/run_seed42_v1

TOOLS EXPOSED:
  1. query_fraud_knowledge   -- query the RAG store for similar events
  2. get_audit_log           -- read recent audit records
  3. get_watchlist           -- read current watchlist.json
  4. get_blocked_wallets     -- read current blocked_wallets.json
  5. get_batch_metrics       -- read batch_history.csv
  6. get_system_state        -- read final_state.json (tau, w)
  7. get_rag_store_stats     -- documents count in RAG store

ZERO BREAKING CHANGES:
  - Reads existing output files -- does NOT write to them
  - Completely separate process from main.py
  - If MCP SDK not installed, this file simply cannot run --
    your main pipeline is 100% unaffected
  - Zero imports in any existing agent or main.py

INSTALL:
  pip install mcp chromadb sentence-transformers

CONNECT FROM CLAUDE DESKTOP (mac):
  Edit ~/Library/Application Support/Claude/claude_desktop_config.json:
  {
    "mcpServers": {
      "fraud-detection": {
        "command": "python",
        "args": [
          "/absolute/path/to/mcp_server.py",
          "--run_dir", "/absolute/path/to/runs/run_seed42_v1"
        ]
      }
    }
  }
"""

import os
import json
import csv
import argparse
import asyncio


# ═══════════════════════════════════════════════════════════════
# SAFE IMPORT
# ═══════════════════════════════════════════════════════════════
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print(
        "MCP SDK not installed.\n"
        "Install with: pip install mcp\n"
        "Your existing pipeline (main.py) is completely unaffected."
    )


# ═══════════════════════════════════════════════════════════════
# HELPERS -- thin readers over your existing output files
# ═══════════════════════════════════════════════════════════════

def _read_json(path: str, default=None):
    if not os.path.exists(path):
        return default if default is not None else {"error": f"{path} not found"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def _read_jsonl_tail(path: str, n: int = 20) -> list:
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        return []
    return records[-n:]


def _read_csv_tail(path: str, n: int = 50) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception:
        return []
    return rows[-n:]


def _query_rag(run_dir: str, query: str, n: int = 3) -> str:
    store_dir = os.path.join(run_dir, "rag_store")
    if not os.path.exists(store_dir):
        return "RAG store not found. Run the pipeline first to build the knowledge base."
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=store_dir)
        col = client.get_collection(name="fraud_events", embedding_function=ef)
        if col.count() == 0:
            return "RAG store is empty. Run the pipeline first to index fraud events."

        results = col.query(
            query_texts=[query],
            n_results=min(n, col.count()),
            where={"y_true": "1"},
            include=["documents", "metadatas", "distances"],
        )
        lines = [f"Top {len(results['documents'][0])} similar confirmed fraud events:\n"]
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), 1):
            lines.append(
                f"[{i}] similarity={1-dist:.3f}\n"
                f"    {doc}\n"
                f"    batch={meta.get('batch','?')} "
                f"action={meta.get('policy_action','?')} "
                f"wallet={str(meta.get('from_address','?'))[:16]}..."
            )
        return "\n".join(lines)
    except ImportError:
        return "chromadb not installed: pip install chromadb sentence-transformers"
    except Exception as e:
        return f"RAG query failed: {e}"


# ═══════════════════════════════════════════════════════════════
# MCP SERVER FACTORY
# ═══════════════════════════════════════════════════════════════

def create_server(run_dir: str) -> "Server":
    server = Server("fraud-detection-mcp")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name="query_fraud_knowledge",
                description=(
                    "Query the RAG knowledge base for historical fraud events "
                    "similar to a given description. Returns top-k confirmed "
                    "fraud cases with similarity scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Transaction description to search for. "
                                           "Example: 'risk_score=0.91 p_rf=0.95 decision=AUTO-BLOCK'",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of similar events to return (default 3).",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_audit_log",
                description=(
                    "Read recent entries from the append-only audit log. "
                    "Each entry contains incident_id, trigger, detection scores, "
                    "policy action, RAG context, and ground truth outcome."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "last_n": {
                            "type": "integer",
                            "description": "Number of most recent records (default 10).",
                            "default": 10,
                        },
                        "trigger_filter": {
                            "type": "string",
                            "description": "Filter: BLOCK, WATCHLIST, batch_summary, or all.",
                            "default": "all",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_watchlist",
                description="Read the current watchlist (wallets under monitoring).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wallet": {
                            "type": "string",
                            "description": "Optional: filter for a specific wallet address.",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_blocked_wallets",
                description="Read the blocked wallet registry with block reasons and counts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wallet": {
                            "type": "string",
                            "description": "Optional: filter for a specific wallet address.",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_batch_metrics",
                description=(
                    "Read per-batch performance metrics: "
                    "precision, recall, F1, ROC-AUC, tau, w."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "last_n": {
                            "type": "integer",
                            "description": "Number of most recent batches (default all).",
                            "default": 100,
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_system_state",
                description=(
                    "Read the current adaptive system state: "
                    "tau_alert, tau_block, fusion weight w, and final run metrics."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="get_rag_store_stats",
                description=(
                    "Return RAG knowledge base statistics: "
                    "total documents indexed and availability."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):

        if name == "query_fraud_knowledge":
            q      = arguments.get("query", "")
            n      = int(arguments.get("n_results", 3))
            result = _query_rag(run_dir, q, n)
            return [types.TextContent(type="text", text=result)]

        elif name == "get_audit_log":
            n       = int(arguments.get("last_n", 10))
            trigger = arguments.get("trigger_filter", "all")
            path    = os.path.join(run_dir, "audit_log.jsonl")
            records = _read_jsonl_tail(path, n=max(n * 5, 50))
            if trigger != "all":
                records = [r for r in records if r.get("trigger") == trigger]
            records = records[-n:]
            return [types.TextContent(type="text", text=json.dumps(records, indent=2, default=str))]

        elif name == "get_watchlist":
            path   = os.path.join(run_dir, "watchlist.json")
            data   = _read_json(path, default={})
            wallet = arguments.get("wallet")
            if wallet:
                data = {wallet: data.get(wallet, "not found")}
            return [types.TextContent(type="text", text=json.dumps(data, indent=2, default=str))]

        elif name == "get_blocked_wallets":
            path   = os.path.join(run_dir, "blocked_wallets.json")
            data   = _read_json(path, default={})
            wallet = arguments.get("wallet")
            if wallet:
                data = {wallet: data.get(wallet, "not found")}
            return [types.TextContent(type="text", text=json.dumps(data, indent=2, default=str))]

        elif name == "get_batch_metrics":
            n    = int(arguments.get("last_n", 100))
            path = os.path.join(run_dir, "batch_history.csv")
            rows = _read_csv_tail(path, n=n)
            return [types.TextContent(type="text", text=json.dumps(rows, indent=2, default=str))]

        elif name == "get_system_state":
            state   = _read_json(os.path.join(run_dir, "final_state.json"))
            summary = _read_json(os.path.join(run_dir, "run_summary.json"), default={})
            result  = {"system_state": state, "final_metrics": summary}
            return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "get_rag_store_stats":
            store_dir = os.path.join(run_dir, "rag_store")
            stats = {"store_dir": store_dir, "exists": os.path.exists(store_dir), "run_dir": run_dir}
            try:
                import chromadb
                from chromadb.utils import embedding_functions
                ef  = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                col = chromadb.PersistentClient(path=store_dir).get_collection("fraud_events", embedding_function=ef)
                stats["total_documents"] = col.count()
                stats["chromadb"]        = "available"
            except ImportError:
                stats["chromadb"]        = "not installed (pip install chromadb sentence-transformers)"
                stats["total_documents"] = 0
            except Exception as e:
                stats["chromadb"]        = f"error: {e}"
                stats["total_documents"] = 0
            return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
async def _main(run_dir: str):
    server = create_server(run_dir)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    if not MCP_AVAILABLE:
        exit(1)
    parser = argparse.ArgumentParser(description="Fraud Detection MCP Server")
    parser.add_argument(
        "--run_dir", type=str, default="runs/run_seed42_v1",
        help="Run directory containing fraud_events.csv, audit_log.jsonl, etc.",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.run_dir))