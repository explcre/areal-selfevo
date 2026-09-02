"""Code as policy: a routing rule written as source, validated, then run as a Router.

The third point on the ``evolve_policy`` axis, alongside ``rule`` (hand-written) and
``learned_weights`` (:class:`~selfevo.routing.contextual.ContextualBanditRouter`). A
weight-based policy can only reweight features it was given; a policy expressed as *code* can
compose them -- thresholds on ratios, nested conditions, cases -- which is a strictly larger
hypothesis class and is what makes this worth a separate arm rather than a variant.

**On safety, stated plainly.** The AST allowlist below is a correctness boundary, not a
security boundary. It exists so a malformed or runaway policy cannot reach outside its
inputs -- no imports, no attribute access, no loops, no comprehensions, no builtins beyond
a numeric handful. It is not a defence against an adversary, and generated policies should
still come from a model you are willing to run code from.

**On termination, also plainly, because the allowlist does not buy it.** Banning loops
bounds the number of steps a policy takes, not the cost of one step: ``9 ** 9 ** 9`` and
``[0] * 10 ** 9`` are allowlisted arithmetic, and neither can be interrupted from Python --
a signal handler runs between bytecodes and each of those is a single opcode. Closing it
would mean dropping ``**`` and ``*``, which ordinary policies use, so it is left open and
named here instead: a policy is bounded in the constructs it may use, never in the time or
memory one construct may take. Run generated policies where a CPU and address-space limit
apply.

**On failure, also plainly.** A policy that raises, loops, or returns something meaningless
falls back to a configured mode AND increments a counter. It never fails silently: an arm
whose policy is broken must not be indistinguishable from an arm whose policy is
conservative.
"""

from __future__ import annotations

import __future__
import ast
import reprlib
from dataclasses import dataclass, field

from selfevo.routing.base import (
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    applicable_modes,
    known_modes,
)

__all__ = ["CodePolicyRouter", "PolicyRejected", "validate_policy_source", "POLICY_SIGNATURE"]

POLICY_SIGNATURE = "def route(features):"

# Node types a decision rule needs, and nothing else. Loops and comprehensions are excluded
# so a policy takes a bounded number of steps; attributes and imports so it cannot reach
# anything it was not handed. Neither bounds the cost of a step -- see the module docstring.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.Name, ast.Store, ast.Load, ast.Constant, ast.Expr,
    ast.If, ast.IfExp, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.Subscript, ast.Call, ast.Tuple, ast.List,
)
_ALLOWED_CALLS = frozenset({"min", "max", "abs", "float", "len", "round"})
_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                  for n in _ALLOWED_CALLS}


class PolicyRejected(ValueError):
    """Raised when policy source fails validation. The message names the offending node."""


def validate_policy_source(source: str) -> ast.Module:
    """Parse and check policy source against the allowlist.

    Args:
        source: Python source defining exactly one function ``route(features)``.

    Returns:
        The parsed module, ready to compile.

    Raises:
        PolicyRejected: On a syntax error, the wrong shape (not exactly one undecorated
            ``route`` function taking one argument and defining nothing inside itself), or
            any construct outside the allowlist. The message names what was rejected,
            because a generated policy is debugged by reading that message.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as exc:
        # Source nested thousands deep ("1+1+1+...") exhausts the parser's stack instead of
        # raising SyntaxError. Uncaught it leaves the constructor as a RecursionError, which
        # is neither a rejection a caller can catch nor a message a policy can be fixed from.
        raise PolicyRejected(f"policy does not parse: {exc}") from exc

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or len(funcs) != 1:
        raise PolicyRejected(
            f"policy must be exactly one function definition and nothing else; found "
            f"{len(tree.body)} top-level statements, {len(funcs)} of them 'def' statements"
        )
    fn = funcs[0]
    if fn.name != "route":
        raise PolicyRejected(f"the function must be named 'route', got {fn.name!r}")
    if fn.decorator_list:
        # A decorator is applied at def time, so it runs inside compile_policy rather than
        # inside route()'s try, and its callee is not reached by the call allowlist below.
        raise PolicyRejected("route must not be decorated")
    a = fn.args
    if len(a.args) != 1 or a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        raise PolicyRejected(f"route must take exactly one positional argument; {POLICY_SIGNATURE}")
    if a.defaults or a.kw_defaults:
        raise PolicyRejected("route must not have default arguments")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise PolicyRejected(
                f"{type(node).__name__} is not allowed in a policy "
                f"(line {getattr(node, 'lineno', '?')}). Allowed: comparisons, arithmetic, "
                f"if/else, subscripting the features dict, and calls to "
                f"{sorted(_ALLOWED_CALLS)}."
            )
        if isinstance(node, ast.FunctionDef) and node is not fn:
            # FunctionDef is allowlisted for ``route`` itself. A nested one cannot be called
            # -- the call allowlist forbids its name -- but its defaults are evaluated when
            # route() runs, which is an expression site none of the checks above cover.
            raise PolicyRejected(
                f"nested function {node.name!r} is not allowed; a policy is one function"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                name = getattr(node.func, "id", type(node.func).__name__)
                raise PolicyRejected(f"call to {name!r} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise PolicyRejected(f"name {node.id!r} is not allowed")
    return tree


def compile_policy(source: str):
    """Validate and compile policy source into a callable.

    Args:
        source: Policy source; see :func:`validate_policy_source`.

    Returns:
        The compiled ``route`` function, whose globals contain only a numeric builtin subset.

    Raises:
        PolicyRejected: If validation fails.
    """
    tree = validate_policy_source(source)
    ns: dict[str, object] = {"__builtins__": dict(_SAFE_BUILTINS)}
    # Annotations stringified, not evaluated: otherwise ``def route(features: 9 ** 9 ** 9)``
    # is a hang at construction, since an annotation is an expression the allowlist accepts
    # and route()'s try never sees. compile() already inherits this flag from this module's
    # own ``from __future__`` line; naming it means the guarantee does not depend on that.
    exec(compile(tree, filename="<policy>", mode="exec",  # noqa: S102 - allowlisted AST
                 flags=__future__.annotations.compiler_flag), ns)
    return ns["route"]


@dataclass
class CodePolicyRouter:
    """A Router whose decision rule is generated source rather than weights or thresholds.

    Args:
        source: Policy source defining ``route(features) -> mode_name``. ``features`` is a
            plain dict: everything in ``ctx.extra`` plus ``solve_rate``, ``group_size`` and
            ``has_target``, so a policy can be written against observability features without
            knowing about this package.
        fallback: Mode used when the policy raises or returns an unusable value. SKIP by
            default: a broken policy should cost nothing rather than train on a guess. It is
            held to the teacher guard too -- see :meth:`_fallback_decision`.
        allowed_modes: Modes the policy may return. ``None`` means every registered mode;
            an empty tuple is rejected rather than read as "all".

    Raises:
        PolicyRejected: If ``source`` fails validation at construction time -- a policy is
            rejected before a run starts, not in the middle of one.
        ValueError: If ``fallback`` or any allowed mode is unregistered, or
            ``allowed_modes`` is empty.
    """

    source: str
    fallback: str = TrainingMode.SKIP
    allowed_modes: tuple[str, ...] | None = None

    errors: int = field(default=0, init=False)
    invalid_returns: int = field(default=0, init=False)
    teacher_blocked: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.fallback not in known_modes():
            raise ValueError(f"unknown fallback mode {self.fallback!r}")
        # ``None`` means every mode an update can be made from -- NOT every registered
        # mode. It used to mean the latter, which put `distill` in the default allowed set:
        # a generated policy returning it was honoured, the unit paid a full rollout, and the
        # run then died inside _compute_advantages. An empty tuple is a caller whose filter
        # returned nothing, and reading it as "all modes" would be the widest possible silent
        # failure, so it is refused rather than defaulted.
        modes = (
            tuple(applicable_modes())
            if self.allowed_modes is None
            else tuple(self.allowed_modes)
        )
        if not modes:
            raise ValueError("allowed_modes must name at least one mode, or be None for all")
        for m in modes:
            if m not in known_modes():
                raise ValueError(f"unknown mode {m!r} in allowed_modes")
        object.__setattr__(self, "allowed_modes", modes)
        self._fn = compile_policy(self.source)

    def _features(self, ctx: RoutingContext) -> dict[str, float]:
        """The dict handed to the policy: observability features plus the basics."""
        f = {k: float(v) for k, v in ctx.extra.items()}
        f["solve_rate"] = float(ctx.solve_rate)
        f["group_size"] = float(ctx.group_size)
        f["has_target"] = 1.0 if ctx.has_target else 0.0
        return f

    def _fallback_decision(self, ctx: RoutingContext, reason: str) -> RoutingDecision:
        """The fallback decision, held to the same teacher guard as the policy's own choice.

        A teacher-requiring ``fallback`` on a unit with no target emits exactly the decision
        :meth:`route`'s guard exists to prevent, and emits it on the rejection path -- where
        the router has already decided the policy was wrong. It degrades to SKIP instead,
        the shape :class:`~selfevo.routing.routers.RandomRouter` uses, and says which.

        Args:
            ctx: The unit being routed, for the target check.
            reason: Why the policy's own answer was not used.

        Returns:
            A one-hot decision on ``fallback``, or on SKIP where ``fallback`` cannot be
            honoured for this unit.
        """
        if known_modes()[self.fallback] and not ctx.has_target:
            return RoutingDecision(
                {TrainingMode.SKIP: 1.0},
                reason=f"{reason}; fallback {self.fallback} needs a target this unit lacks",
            )
        return RoutingDecision({self.fallback: 1.0}, reason=reason)

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Run the policy, and hold it to the same guards every other router obeys.

        Args:
            ctx: The unit to route.

        Returns:
            A decision naming one mode. Every rejection path increments a counter and says
            in ``reason`` what happened, so a broken policy is visible in the log rather
            than mistaken for a conservative one.
        """
        self.calls += 1
        try:
            got = self._fn(self._features(ctx))
        except Exception as exc:  # noqa: BLE001 - any policy failure is the policy's fault
            self.errors += 1
            return self._fallback_decision(ctx, f"code policy raised {type(exc).__name__}")
        if not isinstance(got, str) or got not in (self.allowed_modes or ()):
            self.invalid_returns += 1
            # reprlib, not repr: a policy may return a gigabyte-long str, and this reason is
            # carried into logs. Truncating there would still have built the repr first.
            return self._fallback_decision(
                ctx, f"code policy returned {reprlib.repr(got)}, not an allowed mode"
            )
        if known_modes()[got] and not ctx.has_target:
            # The policy does not get to opt out of the guard every other router obeys.
            self.teacher_blocked += 1
            return self._fallback_decision(ctx, f"code policy chose {got} with no target")
        return RoutingDecision({got: 1.0}, reason="code policy")

    def health(self) -> dict[str, int]:
        """Counts of calls and each rejection path, for logging alongside a run."""
        return {
            "calls": self.calls,
            "errors": self.errors,
            "invalid_returns": self.invalid_returns,
            "teacher_blocked": self.teacher_blocked,
        }
