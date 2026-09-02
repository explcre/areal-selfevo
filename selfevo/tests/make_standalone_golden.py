#!/usr/bin/env python3
"""Record the standalone path's output from the PRE-CHANGE tree, as a fixed-input golden.

WHY A GOLDEN AND NOT AN ASSERTION. The same ``math_bench.run_bench`` produced every headline
number now in the paper, against an ordinary tokenising server. The token-id work must be
additive, and "additive" is a claim about output, so it is recorded as output: this script
runs the fixture in ``test_token_id_eval.py`` against a tokenising stub and writes the whole
results row and every generation to
``selfevo/tests/baselines/standalone_row_pre_token_id.json``.
``test_the_standalone_row_is_byte_identical_to_the_pre_change_baseline`` then regenerates it
and compares.

RUN THIS AGAINST A TREE THAT DOES NOT HAVE THE CHANGE, or the golden records the new
behaviour and the comparison proves nothing:

    rsync -a --exclude .git <pre-change tree>/ /tmp/pre/
    cp selfevo/tests/test_token_id_eval.py /tmp/pre/selfevo/tests/
    MATH_EVAL_DATA=~/evaldata python3 selfevo/tests/make_standalone_golden.py \\
        /tmp/pre <output path>

It refuses to run against a tree whose ``math_bench`` already has the change.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import threading
from http.server import ThreadingHTTPServer


def main() -> int:
    """Record one golden.

    Returns:
        A process exit status.
    """
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    repo = pathlib.Path(sys.argv[1]).resolve()
    out = pathlib.Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "experiments" / "bench"))

    import math_bench as mb

    if hasattr(mb, "server_capabilities"):
        print(
            f"REFUSING: {repo} already carries the token-id change (math_bench has "
            f"server_capabilities). A golden recorded here would record the NEW behaviour and "
            f"the comparison it feeds would be vacuous."
        )
        return 2

    sys.path.insert(0, str(repo / "selfevo" / "tests"))
    import test_token_id_eval as T

    tokdir = T.build_tokenizer_dir(pathlib.Path("/tmp/golden_tok"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), T._Handler)
    srv.policy = T._Policy()
    srv.policy.has_tokenizer = True
    srv.policy.context_limit = 32768
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        import asyncio

        buf = io.StringIO()
        row = asyncio.run(
            mb.run_bench("olympiadbench", T._args(url, tokdir), buf, frozenset({"max_tokens"}))
        )
    finally:
        srv.shutdown()
        srv.server_close()

    payload = {
        "row": T._normalise(row),
        "generations": [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(payload['generations'])} generations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
