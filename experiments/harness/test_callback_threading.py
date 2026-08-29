#!/usr/bin/env python3
"""Reproduce the rollout_complete callback timeout, and show threaded=True fixes it.

AReaL builds the callback receiver as:

    make_server(host, port, app, threaded=False)   # rollout_controller.py:680

With `threaded=False` werkzeug serves ONE request at a time. The trainer fires one
callback per finished rollout; at the published batch_size=256 with n_samples=4 that is
1024 concurrent POSTs. They serialize, later senders exceed their 30s timeout, and the
controller then REJECTS a rollout whose generation had already completed.

This script measures that directly: N concurrent POSTs against the same server
construction, threaded=False vs threaded=True, counting timeouts and wall time.
The handler mimics the real one -- trivial work, but not instantaneous.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, jsonify
from werkzeug.serving import make_server

N_CONCURRENT = 1024   # the real count: batch 256 x n_samples 4
HANDLER_WORK_S = 0.02  # per-request work; the real handler takes a lock and pops a dict
CLIENT_TIMEOUT_S = 5.0


def build(threaded: bool):
    app = Flask(__name__)
    app.logger.disabled = True
    import logging as _l; _l.getLogger("werkzeug").setLevel(_l.ERROR)

    @app.route("/callback/rollout_complete", methods=["POST"])
    def rollout_complete():
        time.sleep(HANDLER_WORK_S)
        return jsonify({"status": "ok"})

    srv = make_server("127.0.0.1", 0, app, threaded=threaded)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, srv.server_port


def run(threaded: bool) -> tuple[int, int, float]:
    srv, port = build(threaded)
    url = f"http://127.0.0.1:{port}/callback/rollout_complete"
    ok = timeouts = 0
    start = time.time()

    def post(i: int) -> bool:
        try:
            r = requests.post(url, json={"task_id": i}, timeout=CLIENT_TIMEOUT_S)
            return r.status_code == 200
        except requests.RequestException:
            return False

    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
        for good in ex.map(post, range(N_CONCURRENT)):
            if good:
                ok += 1
            else:
                timeouts += 1
    elapsed = time.time() - start
    srv.shutdown()
    return ok, timeouts, elapsed


def main() -> None:
    print(f"{N_CONCURRENT} concurrent callbacks, handler={HANDLER_WORK_S}s, "
          f"client timeout={CLIENT_TIMEOUT_S}s\n")
    results = {}
    for threaded in (False, True):
        ok, to, el = run(threaded)
        results[threaded] = (ok, to, el)
        print(f"  threaded={str(threaded):5s}  ok={ok:3d}  timed_out={to:3d}  wall={el:6.2f}s")

    ok_f, to_f, _ = results[False]
    ok_t, to_t, _ = results[True]
    print()
    if to_f > 0 and to_t == 0:
        print(f"REPRODUCED: threaded=False lost {to_f}/{N_CONCURRENT} callbacks; "
              f"threaded=True lost 0.")
    elif to_f == 0:
        print("NOT reproduced at this scale -- raise N_CONCURRENT or HANDLER_WORK_S.")
    else:
        print(f"INCONCLUSIVE: threaded=True still lost {to_t}.")


if __name__ == "__main__":
    main()
