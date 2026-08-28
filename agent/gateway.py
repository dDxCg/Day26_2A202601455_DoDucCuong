"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

THE STARTER'S SHAPE (read this before you start editing `decide()`)
----------------------------------------------------------------------------
This starter FORWARDS ALMOST EVERYTHING AND DENIES NOTHING. That is not a
placeholder oversight — it is the honest zero-defence baseline you are
meant to beat: `bots/rookie` in the kit's own ladder does exactly the same
thing, and RULES.md's own words are "if you cannot beat Rookie you have a
bug, not a strategy." `decide()` below is structured as four named jobs —
ROUTE, ADMIT, AUTHORIZE, BUDGET — each with a one-line TODO naming what a
real implementation checks and why. None of the four currently rejects,
rewrites, or reroutes anything; they are seams, not solutions. Fill them in
using `agent/strategy.py` (routing/budget policy) and `agent/guardrails.py`
(the safety checks) — both already import cleanly from here.

ONE THING WORTH INTERNALISING BEFORE YOU WRITE YOUR FIRST REAL CHECK:
`verdict="deny"` costs the CALLER (your own team) **zero credits** —
CONTRACTS.md 4.1's charging table has exactly one $0 row, and it is this
one. Refusing to make a call you cannot justify is FREE. That makes
abstention a real strategy, not a luxury you can't afford: a `deny` you can
defend beats a `forward` you can't, every time a prosecutor is watching.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.mcp.specs is the tool-economy table (base cost, field weights, is_write,
# needs_lease, deprecation) — same collaborator, same degrade-gracefully rule.
# Used by JOB 2/3/4 below to answer "is this a write", "does this need a live
# lease", "roughly what would this cost" without hand-copying that table here.
try:
    from kit.mcp.specs import TOOL_SPECS, cost as _spec_cost
    _SPECS_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    TOOL_SPECS = {}
    _SPECS_AVAILABLE = False
    _spec_cost = None

# kit.world.anchor.Anchor is pure parsing (no I/O, CONTRACTS.md 1) — safe to
# call from inside decide() itself. `Anchor.parse(...).slug` IS the `path_id`
# for Frame/Deck/Section anchors (kit/world/anchor.py's own path_id() docstring),
# which is what JOB 1 (ROUTE) needs to look up drift — no `World`/`page()` call
# (real file I/O) required at all.
try:
    from kit.world.anchor import Anchor
    _ANCHOR_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    Anchor = None  # type: ignore[assignment, misc]
    _ANCHOR_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from agent.strategy import (
    BudgetPacer,
    cheap_mask,
    is_catalog_trap,
    pick_replica,
    successor_of,
)
from agent.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})

# kit/world/anchor.py NAMESPACES whose slug IS a path_id (the file-identity
# namespaces — day18-style drift is measured per FILE, not per Talk:/Note:/
# Concept: anchor). Mirrors kit/referee/detectors.py's own `_PATH_ID_NAMESPACES`.
_PATH_ID_NAMESPACES: frozenset[str] = frozenset({"Frame", "Deck", "Section"})

# The two write tools kit/mcp/specs.py prices (both `is_write=True` there) —
# used only as a fallback if TOOL_SPECS itself is unavailable (see the
# module-level import guard above); when TOOL_SPECS loads, its own `is_write`
# field is authoritative and this constant is never consulted.
_FALLBACK_WRITE_TOOLS: frozenset[tuple[str, str]] = frozenset(
    {("progress", "record_mastery"), ("content", "flag_stale_slide")}
)


def _is_write_tool(server: str, tool: str) -> bool:
    """CONTRACTS.md's `is_write` flag for `(server, tool)` — JOB 3
    (AUTHORIZE) only applies to writes. `kit.mcp.specs.TOOL_SPECS` first
    (authoritative), `_FALLBACK_WRITE_TOOLS` only if that module never
    imported."""
    if _SPECS_AVAILABLE:
        spec = TOOL_SPECS.get((server, tool))
        if spec is not None:
            return bool(getattr(spec, "is_write", False))
        return False
    return (server, tool) in _FALLBACK_WRITE_TOOLS


def _needs_lease(server: str, tool: str) -> bool:
    """CONTRACTS.md 4.2 mechanic 2: `needs_lease` is reserved for
    `slides.get_frame`. `TOOL_SPECS` first, else the one tool this is ever
    true for."""
    if _SPECS_AVAILABLE:
        spec = TOOL_SPECS.get((server, tool))
        if spec is not None:
            return bool(getattr(spec, "needs_lease", False))
        return False
    return (server, tool) == ("slides", "get_frame")


def _load_drift_map() -> Mapping[str, Mapping[str, Any]]:
    """`path_id -> drift.json record` (`{"drifts": bool, ...}`), read ONCE
    (meant to be called from `Gateway.__init__`, never from `decide()` —
    RULES.md section 3 only forbids I/O *inside* `decide()`; a one-time read
    at construction time is the seam this file uses, same reasoning as
    `agent/strategy.py`'s own module docstring treats `kit.mcp.specs` as
    static reference data). Globs `kit/world/*/drift.json` the same way
    `validate_deck.py`'s `resolve_world()` finds the world directory. Reads
    ONLY `drift.json` (a few KB), never `pages.jsonl` (~12 MB) — this file
    has no need for the full `World` page index, only the drift table.
    Degrades to `{}` (nothing known to drift) if no world is present yet,
    matching every other place in this kit that treats an absent world as
    'ask your instructor', not a crash. `kit/world/<id>/` is the real
    exported corpus, never `truth.json` (never shipped to students) — this
    reads a structural fact (does this path_id's working/canonical replica
    disagree), never an answer key."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = sorted((repo_root / "kit" / "world").glob("*/drift.json"))
    if not candidates:
        return {}
    try:
        with candidates[-1].open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.

    Instance attributes below are this starter's ENTIRE per-duel memory —
    all currently unused by `decide()`'s naive body, but declared here
    (rather than invented ad hoc later) so the four TODO jobs below have
    somewhere obvious to keep state once you implement them. `agent/
    strategy.py` has working building blocks for exactly this (a budget
    pacer, a result cache, a replica-choice heuristic) — this starter does
    not wire them in for you; that wiring is the assignment.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)

        # --- per-duel memory, unused by the naive starter below ---------
        # A cache of anchor -> body-ish data you have already paid for this
        # duel (agent/strategy.py's ResultCache is a ready-made version of
        # this). Populating it needs the *result* of a call, which decide()
        # never sees (it only sees the outgoing Command) — you would fill
        # this from whatever the arena hands back to your agent loop AFTER
        # a call executes, then consult it here on the NEXT decide() call
        # for the same anchor.
        self._seen_anchors: dict[str, Any] = {}
        # Credits you have personally authorised so far this duel — your
        # own running total, independent of (and a cross-check against)
        # `ctx.credits`, which the arena maintains authoritatively.
        self._credits_authorised: int = 0
        # Command ids you have already denied, in case a later job wants to
        # know "have I already said no to this once".
        self._denied_cmd_ids: set[str] = set()
        # JOB 1 (ROUTE): path_id -> drift.json record, loaded ONCE here (see
        # `_load_drift_map`'s own docstring for why this is the right seam —
        # never re-read inside `decide()`).
        self._drift: Mapping[str, Mapping[str, Any]] = _load_drift_map()
        # JOB 4 (BUDGET): this Gateway's own running spend, independent of
        # (and a cross-check against) `ctx.credits` — see
        # `agent/strategy.py`'s `BudgetPacer` for the arithmetic this exists
        # to avoid repeating.
        self._budget_pacer = BudgetPacer(starting_pool=ctx.credits)

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3).
        Raising anything, or returning a `Decision` `__post_init__` rejects,
        is treated by the arena exactly like an explicit deny PLUS a 2
        credit penalty PLUS a scored `integrity` event (CONTRACTS.md 4.1's
        charging table) — so the one thing this method must never do is
        blow up or wander off into I/O, no matter how tempting a "quick
        check" against something external looks. Everything you need to
        decide is already sitting in `cmd` and `self.ctx`.

        This baseline implements all four jobs, each deliberately
        conservative: it only rewrites/denies when it has concrete grounds
        (a measured drift, a missing lease, a target mismatch, a known
        catalog-trap mask against a thin budget) — never a reflex check on
        every call. See agent/README.md's table for why `gateway.py` stays
        the sole enforcement point (`strategy.py`/`guardrails.py` are
        libraries it calls into, never parallel enforcement)."""
        self._telemetry.decision_seen(cmd)

        # ------------------------------------------------------------------
        # JOB 1 — ROUTE: is this the right SERVER/REPLICA for this command?
        # day18-style drift is real and measured (CORPUS-FACTS.md section 2).
        # Two cheap, concrete wins, both REWRITES:
        #   (a) a deprecated tool with a known successor (`slides.search` ->
        #       `slides.query`) — free, removes a `wasteful` hit outright.
        #   (b) an anchor whose path_id is KNOWN to drift (`self._drift`) —
        #       rewrite `mcp-replica` via `agent/strategy.py`'s `pick_replica`.
        #       Only fires on a CONFIRMED drift, never as a blanket default
        #       (agent/prompt.md: not a reflex check every round).
        routed = cmd

        succ = successor_of(routed.server, routed.tool)
        if succ is not None:
            routed = self._rebuild(routed, server=succ[0], tool=succ[1])

        if _ANCHOR_AVAILABLE and self._drift:
            anchor_raw = routed.args.get("anchor") if isinstance(routed.args, dict) else None
            if isinstance(anchor_raw, str):
                try:
                    parsed = Anchor.parse(anchor_raw)
                except Exception:
                    parsed = None
                if parsed is not None and parsed.ns in _PATH_ID_NAMESPACES:
                    path_id = parsed.slug
                    record = self._drift.get(path_id) or {}
                    known_drifting = bool(record.get("drifts", False))
                    if known_drifting:
                        choice = pick_replica(path_id=path_id, known_drifting=True)
                        if routed.headers.get("mcp-replica") != choice.replica:
                            new_headers = dict(routed.headers)
                            new_headers["mcp-replica"] = choice.replica
                            routed = self._rebuild(routed, headers=new_headers)

        # ------------------------------------------------------------------
        # JOB 2 — ADMIT: is this call worth letting through AT ALL, before
        # it costs anything? Both checks below are free abstentions
        # (`verdict="deny"` costs the caller ZERO credits).
        if _needs_lease(routed.server, routed.tool):
            if not routed.lease_id or routed.lease_id not in self.ctx.leases:
                return self.deny(
                    cmd,
                    reason=(
                        f"{routed.server}.{routed.tool} needs a live lease; "
                        f"lease_id={routed.lease_id!r} is not in ctx.leases={self.ctx.leases!r}"
                    ),
                )
        if cmd.cmd_id in self._denied_cmd_ids:
            return self.deny(
                cmd, reason=f"cmd {cmd.cmd_id} was already denied once this duel; nothing has changed"
            )

        # ------------------------------------------------------------------
        # JOB 3 — AUTHORIZE: does `routed` actually belong to WHOM YOU SERVE?
        # `authority_exceeded` (CONTRACTS.md section 6.4) — weight 10, tied
        # heaviest in the rubric. Only applies to writes (`_is_write_tool`);
        # mirrors kit/referee/detectors.py's own `authority_exceeded`
        # detector (read, not imported — that module is the referee's own).
        if _is_write_tool(routed.server, routed.tool):
            args = routed.args if isinstance(routed.args, dict) else {}
            target_learner = args.get("learner")
            if target_learner is None:
                target_learner = args.get("act")
            if isinstance(target_learner, str) and target_learner != self.ctx.act:
                return self.deny(
                    cmd,
                    reason=(
                        f"write {routed.server}.{routed.tool} targets {target_learner!r}, "
                        f"but this duel serves ctx.act={self.ctx.act!r}"
                    ),
                )
            required_scope = f"wiki.write:{routed.server}"
            if required_scope not in self.ctx.scopes:
                return self.deny(
                    cmd,
                    reason=(
                        f"write {routed.server}.{routed.tool} needs scope {required_scope!r}, "
                        f"not in ctx.scopes={sorted(self.ctx.scopes)!r}"
                    ),
                )

        # ------------------------------------------------------------------
        # JOB 4 — BUDGET: can the DUEL (all 10 rounds, not just this call)
        # actually afford `routed` as written? `fields=("*",)`/no mask on a
        # catalog-trap tool (`registry.list_servers`, `glossary.list_terms`)
        # is a "punishment button" (FINAL-PLAN.md 4.1) — deny it outright
        # once the pacer's reserve floor is at risk, rather than guessing a
        # cheap mask on the model's behalf (this job has no visibility into
        # what the answer will actually cite, so it can only refuse, never
        # invent a `cheap_mask`).
        estimated_cost = 0
        if _spec_cost is not None:
            try:
                estimated_cost = _spec_cost(routed.server, routed.tool, fields=routed.fields, n_rows=1)
            except KeyError:
                estimated_cost = 0
        if is_catalog_trap(routed.server, routed.tool, routed.fields) and not self._budget_pacer.is_affordable(
            self.ctx.round, estimated_cost
        ):
            return self.deny(
                cmd,
                reason=(
                    f"{routed.server}.{routed.tool} is a catalog-trap call (default/full mask) "
                    f"and would breach the reserve floor at round {self.ctx.round}"
                ),
            )

        call = self._to_tool_call(routed)
        verdict = "forward" if routed is cmd else "rewrite"
        decision = Decision(verdict=verdict, call=call)
        self._budget_pacer.record_spend(self.ctx.round, estimated_cost)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Not called anywhere in this starter's `decide()` — a ready-made
        helper for when you fill in JOB 2 / JOB 3 above, so denying doesn't
        mean hand-building a `Decision` inline at every call site. Kept as
        a real method (not a stub) because the shape of a correct denial —
        no `call`, a non-empty `reason` — is exactly the thing worth
        getting right by construction rather than by convention."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    @staticmethod
    def _rebuild(cmd: Command, **overrides: Any) -> Command:
        """`Command` is frozen (CONTRACTS.md 4.1) — JOB 1 (ROUTE) needs to
        change `server`/`tool`/`headers` on a `rewrite` verdict, so this
        reconstructs a new `Command` with the given fields overridden,
        everything else copied verbatim (still going through
        `Command.__post_init__`'s own validation)."""
        fields: dict[str, Any] = {
            "cmd_id": cmd.cmd_id,
            "kind": cmd.kind,
            "raw": cmd.raw,
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        fields.update(overrides)
        return Command(**fields)

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — the naive starter forwards everything ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=("lse_7f21",),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
        assert decision.verdict == "forward"
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        assert tuple(call_dict["fields"]) == cmd.fields

    print(f"\n=== Gateway.deny — the unused-by-default free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
