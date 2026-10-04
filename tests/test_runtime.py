"""Exercise the unmodified manager with isolated fake services; no GPU required."""

import concurrent.futures
import http.server
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
TEXT = "model-fleet-qwen27-long.service"
TEXT_B = "model-fleet-qwen35-long.service"
MEDIA = "model-fleet-comfy.service"

SYSTEMCTL = r'''
import json, os, pathlib, sys
root = pathlib.Path(os.environ["FAKE_FLEET_ROOT"])
args = [a for a in sys.argv[1:] if a != "--user"]
if len(args) != 2 or args[0] not in ("is-active", "start", "stop"):
    sys.exit(2)
verb, unit = args
known = {"model-fleet-qwen27-long.service", "model-fleet-qwen35-long.service",
         "model-fleet-comfy.service", "model-fleet-proxy.service",
         "model-fleet-g9v3-media.service"}
if unit not in known:
    sys.exit(2)
state_file = root / "units.json"
state = json.loads(state_file.read_text())
if verb == "is-active":
    print("active" if state.get(unit) else "inactive")
    sys.exit(0 if state.get(unit) else 3)
with (root / "events.jsonl").open("a") as f:
    f.write(json.dumps([verb, unit]) + "\n")
if verb == "start" and (root / "fail-unit").read_text() == unit:
    print("simulated worker start failure", file=sys.stderr)
    sys.exit(1)
state[unit] = verb == "start"
tmp = root / "units.tmp"
tmp.write_text(json.dumps(state))
tmp.replace(state_file)
'''


@unittest.skipUnless(NODE, "Node is required")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fleet-recipe-test-")
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        (self.tmp / "units.json").write_text("{}")
        (self.tmp / "events.jsonl").write_text("")
        (self.tmp / "fail-unit").write_text("")
        for name, body in {
            "systemctl": SYSTEMCTL,
            "nvidia-smi": 'print("0, GPU-EXAMPLE, Fake GPU, 24576, 0, 24576, 0")',
        }.items():
            shim = self.bin / name
            shim.write_text(f"#!{sys.executable}\n{body}\n")
            shim.chmod(0o700)

        self.ready = threading.Event()
        self.ready.set()
        case = self

        class Health(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                units = json.loads((case.tmp / "units.json").read_text())
                if self.path == "/text/health":
                    live = units.get(TEXT) or units.get(TEXT_B)
                elif self.path == "/comfy/system_stats":
                    live = units.get(MEDIA)
                else:
                    live = False
                code = 200 if live and case.ready.is_set() else 503
                body = json.dumps({"ready": code == 200}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.health = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
        self.addCleanup(self.health.server_close)
        self.addCleanup(self.health.shutdown)
        threading.Thread(target=self.health.serve_forever, daemon=True).start()
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            manager_port = port.getsockname()[1]
        health_url = f"http://127.0.0.1:{self.health.server_port}"
        self.url = f"http://127.0.0.1:{manager_port}"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("MODEL_FLEET_")}
        self.env.update({
            "PATH": str(self.bin),
            "FAKE_FLEET_ROOT": str(self.tmp),
            "MODEL_FLEET_HOST": "127.0.0.1",
            "MODEL_FLEET_PORT": str(manager_port),
            "MODEL_FLEET_URL": self.url,
            "MODEL_FLEET_DATA_ROOT": str(self.tmp / "data"),
            "MODEL_FLEET_STATE_PATH": str(self.tmp / "data/state.json"),
            "MODEL_FLEET_AVAILABLE_PROFILES": "qwen27-long,qwen35-long,media,idle",
            "MODEL_FLEET_MEDIA_CONTINUITY": "0",
            "MODEL_FLEET_MEDIA_PRELOAD": "0",
            "MODEL_FLEET_STICKY_MEDIA": "0",
            "MODEL_FLEET_TEXT_HEALTH_URL": health_url + "/text/health",
            "MODEL_FLEET_COMFY_HEALTH_URL": health_url + "/comfy/system_stats",
        })
        self.log = (self.tmp / "manager.log").open("w+")
        self.addCleanup(self.log.close)
        self.proc = subprocess.Popen(
            [NODE, str(ROOT / "runtime/runtime.mjs")],
            env=self.env, stdout=self.log, stderr=subprocess.STDOUT,
        )
        self.addCleanup(self.stop_manager)
        for _ in range(100):
            try:
                if self.call("/health")[0] == 200:
                    return
            except OSError:
                pass
            if self.proc.poll() is not None:
                self.log.seek(0)
                self.fail(self.log.read())
            time.sleep(0.05)
        self.fail("manager did not start")

    def stop_manager(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=20)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def transition(self, target, **extra):
        status, data = self.call("/transition", {"target": target, **extra})
        self.assertEqual(status, 200, data)
        self.assertEqual((data["target"], data["phase"]), (target, "ready"))
        return data

    def events(self):
        return [json.loads(line) for line in (self.tmp / "events.jsonl").read_text().splitlines()]

    def test_text_media_text_order_and_idle(self):
        self.transition("qwen27-long")
        self.transition("media")
        self.transition("qwen35-long")
        self.transition("idle")
        self.assertEqual(self.events(), [
            ["start", TEXT], ["stop", TEXT], ["start", MEDIA],
            ["stop", MEDIA], ["start", TEXT_B], ["stop", TEXT_B],
        ])
        self.assertFalse(any(json.loads((self.tmp / "units.json").read_text()).values()))

    def test_lease_rejects_competing_owner(self):
        self.assertEqual(self.call("/lease/acquire", {"owner": "job-a"})[0], 200)
        self.assertEqual(self.call("/lease/acquire", {"owner": "job-b"})[0], 409)
        self.assertEqual(self.call("/transition", {"target": "media", "owner": "job-b"})[0], 409)
        self.assertEqual(self.call("/transition", {"target": "media"})[0], 409)
        self.assertEqual(self.call("/lease/release", {"owner": "job-b"})[0], 409)
        self.assertEqual(self.events(), [])
        self.transition("qwen27-long", owner="job-a")
        self.assertEqual(self.call("/lease/release", {"owner": "job-a"})[0], 200)
        self.transition("idle")

    def test_concurrent_transitions_wait_for_readiness(self):
        self.ready.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            try:
                first = pool.submit(self.call, "/transition", {"target": "qwen27-long"})
                deadline = time.monotonic() + 5
                while ["start", TEXT] not in self.events() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertIn(["start", TEXT], self.events())
                second = pool.submit(self.call, "/transition", {"target": "media"})
                time.sleep(0.2)
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                self.assertNotIn(["start", MEDIA], self.events())
                state = json.loads((self.tmp / "data/state.json").read_text())
                self.assertEqual(state["phase"], "transitioning")
            finally:
                self.ready.set()
            self.assertEqual(first.result(timeout=20)[0], 200)
            self.assertEqual(second.result(timeout=20)[0], 200)
        self.assertEqual(self.events(), [["start", TEXT], ["stop", TEXT], ["start", MEDIA]])

    def test_failed_start_records_error_without_automatic_rollback(self):
        self.transition("qwen27-long")
        (self.tmp / "fail-unit").write_text(MEDIA)
        status, data = self.call("/transition", {"target": "media"})
        self.assertEqual(status, 500, data)
        self.assertEqual(self.call("/status")[1]["phase"], "error")
        self.assertFalse(any(json.loads((self.tmp / "units.json").read_text()).values()))
        (self.tmp / "fail-unit").write_text("")
        self.transition("qwen27-long")

    def test_status_detects_a_lost_worker(self):
        self.transition("qwen27-long")
        (self.tmp / "units.json").write_text("{}")
        self.assertEqual(self.call("/status")[1]["phase"], "stale")

    def test_mcp_bridge_profiles_and_idle_transition(self):
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "list_runtime_profiles"}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "activate_runtime_profile", "arguments": {"target": "idle"}}},
        ]
        result = subprocess.run(
            [NODE, str(ROOT / "runtime/runtime-mcp.mjs")],
            input="".join(json.dumps(m) + "\n" for m in messages),
            env=self.env, text=True, capture_output=True, timeout=20, check=True,
        )
        replies = {m["id"]: m for m in map(json.loads, result.stdout.splitlines())}
        self.assertEqual(set(replies), {1, 2, 3, 4})
        self.assertTrue(all("result" in m for m in replies.values()), replies)
        self.assertEqual(len(replies[2]["result"]["tools"]), 3)
        self.assertIn("media", replies[3]["result"]["structuredContent"])
        self.assertEqual(replies[4]["result"]["structuredContent"]["phase"], "ready")


if __name__ == "__main__":
    unittest.main()
