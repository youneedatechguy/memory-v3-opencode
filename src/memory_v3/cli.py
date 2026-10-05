"""
memory-v3 -- CLI wrapper
Command-line interface for the memory-v3 system.
Same pattern as v2 but with new commands for v3 features.
"""

import argparse
import json
import sys


def _get_conn():
    """Get a database connection."""
    from .db import init_db

    return init_db()


def _pp(obj):
    """Pretty-print JSON to stdout."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            print(obj)
            return
    print(json.dumps(obj, indent=2, default=str))


def _as_obj(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return obj
    return obj


def _pp_exit_on_failure(obj):
    parsed = _as_obj(obj)
    _pp(parsed)
    if isinstance(parsed, dict) and (parsed.get("error") or parsed.get("errors", 0)):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_stats(args):
    """Show memory system statistics."""
    from .server import stats

    _pp(stats())


def cmd_add(args):
    """Add a new memory."""
    tags = args.tags.split(",") if args.tags else None
    from .server import add_memory

    result = add_memory(
        content=args.content,
        content_type=args.type,
        tags=tags,
        source=args.source,
        protected=args.protected,
        author=args.author,
        confidence=args.confidence,
        governance_layer=args.layer,
        structure_type=args.structure,
    )
    _pp(result)


def cmd_search(args):
    """Search memories."""
    from .server import search

    result = search(query=args.query, limit=args.limit, content_type=args.type)
    _pp(result)


def cmd_keyword(args):
    """Keyword search (BM25)."""
    from .server import keyword_search

    result = keyword_search(keywords=args.keywords, limit=args.limit)
    _pp(result)


def cmd_get(args):
    """Get a single memory by ID."""
    from .server import get

    result = get(memory_id=args.id)
    _pp(result)


def cmd_update(args):
    """Update an existing memory."""
    tags = args.tags.split(",") if args.tags else None
    from .server import update

    result = update(
        memory_id=args.id,
        content=args.content,
        tags=tags,
        protected=args.protected,
        governance_layer=args.layer,
        structure_type=args.structure,
    )
    _pp(result)


def cmd_forget(args):
    """Archive a memory."""
    from .server import forget

    result = forget(memory_id=args.id, reason=args.reason)
    _pp(result)


def cmd_recent(args):
    """List recent memories."""
    from .server import list_recent

    result = list_recent(hours=args.hours, limit=args.limit)
    _pp(result)


def cmd_topics(args):
    """List topic clusters."""
    from .server import list_topics

    result = list_topics(limit=args.limit)
    _pp(result)


def _cmd_reembed(args):
    message = (
        "memory-v3 reindex --reembed is disabled; use the explicit "
        "shadow migration/reindex workflow"
    )
    _pp(
        {
            "error": message,
            "code": "reembed_disabled",
            "status": "disabled",
        }
    )
    sys.exit(1)


def cmd_reindex(args):
    """Re-index the vault."""
    if hasattr(args, "reembed") and args.reembed:
        _cmd_reembed(args)
        return
    from .server import reindex

    result = reindex(force=args.force)
    _pp_exit_on_failure(result)


def cmd_handoffs(args):
    """Manage active handoffs."""
    if args.handoff_action == "list":
        from .server import list_active_handoffs

        result = list_active_handoffs(agent_id=args.agent_id, since=args.since)
        _pp_exit_on_failure(result)
        return
    if args.handoff_action == "sweep":
        from .server import sweep_expired_handoffs

        result = sweep_expired_handoffs(now=args.now)
        _pp_exit_on_failure(result)
        return
    raise ValueError(f"Unknown handoff_action: {args.handoff_action}")


def cmd_integrity(args):
    """Check vault integrity."""
    from .server import check_integrity

    result = check_integrity()
    _pp(result)


def cmd_extract(args):
    """Extract facts from conversation text."""
    # Read from stdin if no text provided
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    from .server import extract_from_conversation

    result = extract_from_conversation(text=text, source=args.source, author=args.author)
    _pp(result)


def cmd_auto_extract(args):
    """Auto-extract memory from session boundary triggers."""

    from .auto_extract.runner import AutoExtractRunner

    runner = AutoExtractRunner()

    if args.auto_action == "run":
        res = runner.run(
            trigger_type=args.trigger,
            fingerprint=args.fingerprint,
            dry_run=args.dry_run,
            force=args.force,
            git_dir=args.git_dir,
            work_tree=args.work_tree,
            text_override=args.text,
            vault_dir=args.vault_dir,
            since_days=args.since_days,
            max_vault_files=args.max_vault_files,
            max_chars_per_file=args.max_chars_per_file,
        )
        _pp(
            {
                "trigger_type": res.trigger_type,
                "fingerprint": res.fingerprint,
                "collected_chars": res.collected_chars,
                "extracted": res.extracted,
                "skipped": res.skipped,
                "reason": res.reason,
            }
        )
        return

    if args.auto_action == "status":
        entry = runner.state.get(args.trigger, args.fingerprint)
        if not entry:
            _pp({"status": "not_found", "trigger": args.trigger, "fingerprint": args.fingerprint})
            return
        _pp(
            {
                "status": entry.status,
                "trigger": args.trigger,
                "fingerprint": entry.fingerprint,
                "last_run_ts": entry.last_run_ts,
            }
        )
        return

    if args.auto_action == "print-git-hook":
        print(
            "#!/bin/sh\n"
            "set -e\n"
            'MEMORY_V3_BIN="$(command -v memory-v3)"\n'
            'if [ -z "$MEMORY_V3_BIN" ]; then\n'
            "  echo 'memory-v3 not found on PATH' >&2\n"
            "  exit 1\n"
            "fi\n"
            'GIT_DIR_ESCAPED="${GIT_DIR:-}"\n'
            'GIT_WORK_TREE_ESCAPED="${GIT_WORK_TREE:-}"\n'
            'if [ -z "$GIT_WORK_TREE_ESCAPED" ]; then\n'
            '  GIT_WORK_TREE_ESCAPED="$(pwd)"\n'
            "fi\n"
            'if [ -z "$GIT_DIR_ESCAPED" ]; then\n'
            '  GIT_DIR_ESCAPED="$(git rev-parse --git-dir)"\n'
            "fi\n"
            'FINGERPRINT=$(git --git-dir "$GIT_DIR_ESCAPED" '
            '--work-tree "$GIT_WORK_TREE_ESCAPED" rev-parse HEAD)\n'
            '"$MEMORY_V3_BIN" auto-extract run --trigger git '
            '--fingerprint "$FINGERPRINT" --git-dir "$GIT_DIR_ESCAPED" '
            '--work-tree "$GIT_WORK_TREE_ESCAPED"\n'
        )
        return

    raise ValueError(f"Unknown auto_action: {args.auto_action}")


def cmd_compact(args):
    """Compact text."""
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    from .server import compact_text

    result = compact_text(text=text, max_ratio=args.ratio, verify=not args.no_verify)
    _pp(result)


def cmd_graph_search(args):
    """Search the knowledge graph."""
    from .server import graph_search

    result = graph_search(query=args.query, top_k=args.limit)
    _pp(result)


def cmd_graph_stats(args):
    """Show knowledge graph statistics."""
    from .server import graph_stats_tool

    result = graph_stats_tool()
    _pp(result)


def cmd_decay(args):
    """Run decay sweep."""
    from .server import decay_sweep

    result = decay_sweep()
    _pp(result)


def cmd_sync(args):
    """Sync agent changelog."""
    from .server import agent_sync

    result = agent_sync(agent_id=args.agent_id)
    _pp(result)


def cmd_conflicts(args):
    """Check for memory conflicts."""
    from .server import check_conflicts

    result = check_conflicts()
    _pp(result)


# --- NEW v3 commands ---


def cmd_consolidate(args):
    """Run sleep consolidation cycle."""
    from .server import sleep_consolidation

    result = sleep_consolidation(cycle_type=args.cycle)
    _pp(result)


def cmd_verify(args):
    """Verify memory source integrity (HaluMem)."""
    ids = None
    if args.ids:
        ids = [int(x.strip()) for x in args.ids.split(",")]
    from .server import verify_memories

    result = verify_memories(memory_ids=ids, max_check=args.max_check)
    _pp(result)


def cmd_set_layer(args):
    """Change governance layer of a memory."""
    from .server import set_governance_layer

    result = set_governance_layer(memory_id=args.id, layer=args.layer)
    _pp(result)


def cmd_causal_chain(args):
    """Trace a causal chain."""
    from .server import get_causal_chain

    result = get_causal_chain(query=args.query, max_depth=args.depth)
    _pp(result)


def cmd_timeline(args):
    """Get temporal timeline."""
    from .server import get_timeline

    result = get_timeline(
        query=args.query,
        start_date=args.start,
        end_date=args.end,
        limit=args.limit,
    )
    _pp(result)


def cmd_link(args):
    """Link two memories."""
    from .server import link_memories

    result = link_memories(
        memory_id_a=args.id_a,
        memory_id_b=args.id_b,
        description=args.description,
    )
    _pp(result)


def cmd_build_kg(args):
    """Deprecated graph rebuild command retained for parser compatibility."""
    _pp(
        {
            "error": "memory-v3 build-kg is deprecated and disabled; graph artifacts are dormant",
            "code": "build_kg_disabled",
            "status": "disabled",
        }
    )
    sys.exit(1)


def cmd_serve(args):
    """Run the MCP server."""
    from .server import run

    run()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="memory-v3",
        description="memory-v3: Next-generation brain-inspired memory system CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- stats ---
    sub.add_parser("stats", help="Show memory system statistics")

    # --- add ---
    p = sub.add_parser("add", help="Add a new memory")
    p.add_argument("content", help="Memory content text")
    p.add_argument("-t", "--type", default="fact", help="Content type (default: fact)")
    p.add_argument("--tags", default=None, help="Comma-separated tags")
    p.add_argument("--source", default=None, help="Source file path")
    p.add_argument("--protected", action="store_true", help="Mark as protected")
    p.add_argument("--author", default="claude-code", help="Author (default: claude-code)")
    p.add_argument("--confidence", type=float, default=0.95, help="Confidence (default: 0.95)")
    p.add_argument("--layer", type=int, default=None, help="Governance layer (1-4)")
    p.add_argument("--structure", default=None, help="Structure type")

    # --- search ---
    p = sub.add_parser("search", help="Hybrid search (BM25 + vector + ACT-R)")
    p.add_argument("query", help="Search query")
    p.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    p.add_argument("-t", "--type", default=None, help="Filter by content type")

    # --- keyword ---
    p = sub.add_parser("keyword", help="Pure BM25 keyword search")
    p.add_argument("keywords", help="Search keywords")
    p.add_argument("-n", "--limit", type=int, default=10, help="Max results")

    # --- get ---
    p = sub.add_parser("get", help="Get a memory by ID")
    p.add_argument("id", type=int, help="Memory ID")

    # --- update ---
    p = sub.add_parser("update", help="Update a memory")
    p.add_argument("id", type=int, help="Memory ID")
    p.add_argument("--content", default=None, help="New content")
    p.add_argument("--tags", default=None, help="New tags (comma-separated)")
    p.add_argument("--protected", type=bool, default=None, help="Protected status")
    p.add_argument("--layer", type=int, default=None, help="Governance layer (1-4)")
    p.add_argument("--structure", default=None, help="Structure type")

    # --- forget ---
    p = sub.add_parser("forget", help="Archive a memory")
    p.add_argument("id", type=int, help="Memory ID")
    p.add_argument("--reason", default="manual", help="Reason for archiving")

    # --- recent ---
    p = sub.add_parser("recent", help="List recent memories")
    p.add_argument("--hours", type=int, default=24, help="Hours to look back")
    p.add_argument("-n", "--limit", type=int, default=20, help="Max results")

    # --- topics ---
    p = sub.add_parser("topics", help="List topic clusters")
    p.add_argument("-n", "--limit", type=int, default=20, help="Max topics")

    # --- reindex ---
    p = sub.add_parser("reindex", help="Re-index the vault")
    p.add_argument("--force", action="store_true", help="Force full re-index")
    p.add_argument(
        "--reembed",
        action="store_true",
        help="Disabled; use the explicit shadow migration/reindex workflow",
    )

    # --- handoffs ---
    p = sub.add_parser("handoffs", help="List or sweep temporary handoffs")
    handoff_sub = p.add_subparsers(dest="handoff_action", required=True)
    handoff_list = handoff_sub.add_parser("list", help="List active handoffs")
    handoff_list.add_argument("--agent-id", default=None, help="Filter by agent/author")
    handoff_list.add_argument("--since", default=None, help="Only handoffs updated since ISO time")
    handoff_sweep = handoff_sub.add_parser("sweep", help="Delete expired handoffs if enabled")
    handoff_sweep.add_argument("--now", default=None, help="Override current time for sweeping")

    # --- integrity ---
    sub.add_parser("integrity", help="Check vault integrity and staleness")

    # --- extract ---
    p = sub.add_parser("extract", help="Extract facts from conversation text")
    p.add_argument("--text", default=None, help="Text to extract from (or pipe stdin)")
    p.add_argument("--source", default=None, help="Source label")
    p.add_argument("--author", default="claude-code", help="Author")

    # --- compact ---
    p = sub.add_parser("compact", help="Compact text via CogCanvas pipeline")
    p.add_argument("--text", default=None, help="Text to compact (or pipe stdin)")
    p.add_argument("--ratio", type=float, default=3.0, help="Max compression ratio")
    p.add_argument("--no-verify", action="store_true", help="Skip verification")

    # --- auto-extract ---
    p = sub.add_parser("auto-extract", help="Auto-extract memory at session boundaries")
    auto_sub = p.add_subparsers(dest="auto_action", required=True)

    run = auto_sub.add_parser("run", help="Run auto-extract for a trigger")
    run.add_argument(
        "--trigger",
        required=True,
        choices=["git", "git-wip", "daily", "vault"],
        help="Trigger type",
    )
    run.add_argument(
        "--fingerprint",
        default="auto",
        help="Run fingerprint (default: auto)",
    )
    run.add_argument("--dry-run", action="store_true", help="Collect only; do not extract")
    run.add_argument("--force", action="store_true", help="Re-run even if already processed")
    run.add_argument("--text", default=None, help="Override extracted text (used for shell/editor)")
    run.add_argument("--git-dir", default=None, help="Path to .git directory for git trigger")
    run.add_argument("--work-tree", default=None, help="Work tree path for git trigger")
    run.add_argument("--vault-dir", default=None, help="Vault directory")
    run.add_argument("--since-days", type=int, default=1, help="Days back to scan")
    run.add_argument("--max-vault-files", type=int, default=50, help="Max md files")
    run.add_argument(
        "--max-chars-per-file",
        type=int,
        default=40000,
        help="Max characters to read per markdown file",
    )

    status = auto_sub.add_parser("status", help="Check auto-extract state for a fingerprint")
    status.add_argument("--trigger", required=True, choices=["git", "git-wip", "daily", "vault"])
    status.add_argument("--fingerprint", required=True)

    auto_sub.add_parser("print-git-hook", help="Print a post-commit hook script")

    # --- graph ---
    p = sub.add_parser("graph", help="Search knowledge graph")
    p.add_argument("query", help="Search query")
    p.add_argument("-n", "--limit", type=int, default=10, help="Max results")

    # --- graph-stats ---
    sub.add_parser("graph-stats", help="Knowledge graph statistics")

    # --- decay ---
    sub.add_parser("decay", help="Run FadeMem decay sweep")

    # --- sync ---
    p = sub.add_parser("sync", help="Sync agent changelog")
    p.add_argument("agent_id", help="Agent identifier")

    # --- conflicts ---
    sub.add_parser("conflicts", help="Check for memory conflicts")

    # --- consolidate (NEW) ---
    p = sub.add_parser("consolidate", help="Run sleep consolidation cycle")
    p.add_argument(
        "--cycle",
        default="full",
        choices=["replay", "reorganize", "compress", "prune", "full"],
        help="Cycle type (default: full)",
    )

    # --- verify (NEW) ---
    p = sub.add_parser("verify", help="Verify memory source integrity (HaluMem)")
    p.add_argument("--ids", default=None, help="Comma-separated memory IDs")
    p.add_argument("--max-check", type=int, default=50, help="Max memories to check")

    # --- set-layer (NEW) ---
    p = sub.add_parser("set-layer", help="Change governance layer of a memory")
    p.add_argument("id", type=int, help="Memory ID")
    p.add_argument("layer", type=int, choices=[1, 2, 3, 4], help="New governance layer")

    # --- causal-chain (NEW) ---
    p = sub.add_parser("causal-chain", help="Trace a causal chain")
    p.add_argument("query", help="Search query for causal start node")
    p.add_argument("--depth", type=int, default=5, help="Max chain depth")

    # --- timeline (NEW) ---
    p = sub.add_parser("timeline", help="Get temporal timeline")
    p.add_argument("query", help="Search query for events")
    p.add_argument("--start", default=None, help="Start date (ISO 8601)")
    p.add_argument("--end", default=None, help="End date (ISO 8601)")
    p.add_argument("-n", "--limit", type=int, default=20, help="Max events")

    # --- link (NEW) ---
    p = sub.add_parser("link", help="Link two memories")
    p.add_argument("id_a", type=int, help="First memory ID")
    p.add_argument("id_b", type=int, help="Second memory ID")
    p.add_argument("--description", default="", help="Link description")

    # --- build-kg ---
    p = sub.add_parser(
        "build-kg",
        help="Deprecated disabled graph build command",
    )
    p.add_argument("--force", action="store_true", help="Deprecated; ignored")
    p.add_argument(
        "--backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="Deprecated; ignored. build-kg is disabled",
    )

    # --- serve ---
    sub.add_parser("serve", help="Run the MCP server")

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "stats": cmd_stats,
    "add": cmd_add,
    "search": cmd_search,
    "keyword": cmd_keyword,
    "get": cmd_get,
    "update": cmd_update,
    "forget": cmd_forget,
    "recent": cmd_recent,
    "topics": cmd_topics,
    "reindex": cmd_reindex,
    "handoffs": cmd_handoffs,
    "integrity": cmd_integrity,
    "extract": cmd_extract,
    "auto-extract": cmd_auto_extract,
    "compact": cmd_compact,
    "graph": cmd_graph_search,
    "graph-stats": cmd_graph_stats,
    "decay": cmd_decay,
    "sync": cmd_sync,
    "conflicts": cmd_conflicts,
    "consolidate": cmd_consolidate,
    "verify": cmd_verify,
    "set-layer": cmd_set_layer,
    "causal-chain": cmd_causal_chain,
    "timeline": cmd_timeline,
    "link": cmd_link,
    "build-kg": cmd_build_kg,
    "serve": cmd_serve,
}


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
