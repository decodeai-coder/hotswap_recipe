#!/usr/bin/env python3
"""Opt-in hardware test. Creates temporary user units and runs real inference.

Unlike test_runtime.py, this script uses real systemd and an NVIDIA GPU. It
requires explicit model/binary paths and an otherwise unused set of Fleet user
service names. It never downloads models or stops pre-existing services.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import select
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


WORKERS = ("model-fleet-qwen27-long.service", "model-fleet-qwen35-long.service")
MANAGER = "fleet-hot-swap-validation-manager.service"
KNOWN_UNITS = (*WORKERS, "model-fleet-comfy.service",
               "model-fleet-g9v3-media.service", "model-fleet-proxy.service", MANAGER)


def run(*args, check=True, env=None):
    return subprocess.run(args, check=check, env=env, text=True, capture_output=True, timeout=40)


def http(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def properties(unit):
    result = run("systemctl", "--user", "show", unit,
                 "-p", "LoadState", "-p", "ActiveState", "-p", "MainPID",
                 "-p", "ExecMainStartTimestampMonotonic",
                 "-p", "ExecMainExitTimestampMonotonic", check=False)
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if "LoadState" not in values:
        raise RuntimeError("Cannot inspect user service state: " + result.stderr.strip())
    return values


def gpu_snapshot(gpu_uuid):
    fields = run("nvidia-smi", "--id=" + gpu_uuid,
                 "--query-gpu=name,driver_version,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits").stdout.strip().split(",")
    processes = {}
    for line in run("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                    "--format=csv,noheader,nounits").stdout.splitlines():
        gpu, pid, memory = [v.strip() for v in line.split(",")]
        if gpu == gpu_uuid:
            processes[int(pid)] = int(memory)
    return {
        "name": fields[0].strip(), "driver": fields[1].strip(),
        "used_mib": int(fields[2]), "free_mib": int(fields[3]),
        "utilization_percent": int(fields[4]), "processes": processes,
    }


def observe_exit(pid):
    """Observe actual process exit even if systemd garbage-collects unit history."""
    fd = os.pidfd_open(pid)
    observation = {}

    def wait():
        try:
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            events = poller.poll(185000)
            if any(mask & select.POLLIN for _, mask in events):
                observation["exit_observed_us"] = time.monotonic_ns() // 1000
        finally:
            os.close(fd)

    watcher = threading.Thread(target=wait, daemon=True)
    watcher.start()
    return observation, watcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gpu-uuid", "model-a", "model-b", "llama-server", "node", "runtime", "work-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--manager-port", type=int, default=18199)
    parser.add_argument("--text-port", type=int, default=18091)
    args = parser.parse_args()
    for value in (args.model_a, args.model_b, args.llama_server, args.node, args.runtime, args.work_dir):
        if not Path(value).is_absolute() or re.search(r'[\s%$"\\]', value):
            parser.error("Use absolute paths without whitespace or systemd expansion characters")
    if not re.fullmatch(r"GPU-[a-fA-F0-9-]+", args.gpu_uuid):
        parser.error("A real GPU UUID is required")
    if args.manager_port == args.text_port:
        parser.error("Manager and worker ports must differ")
    for port in (args.manager_port, args.text_port):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    for unit in KNOWN_UNITS:
        if properties(unit)["LoadState"] != "not-found":
            raise RuntimeError(f"Refusing to touch an existing user unit: {unit}")

    before = gpu_snapshot(args.gpu_uuid)
    if before["free_mib"] < 8192:
        raise RuntimeError("This small-model test requires at least 8 GiB free")
    # This helper is deliberately bounded to small models; it is not a scheduler.
    for model in (args.model_a, args.model_b):
        if not 0 < Path(model).stat().st_size <= 4 * 1024**3:
            raise RuntimeError("Each test GGUF must be no larger than 4 GiB")
    devices = run(args.llama_server, "--list-devices",
                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid))
    if "CUDA0:" not in devices.stdout + devices.stderr:
        raise RuntimeError("Selected llama-server does not expose the requested CUDA device")

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    output = work / "result.json"
    if output.exists():
        raise RuntimeError("Use a fresh result directory; evidence will not be overwritten")
    unit_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    owned = []
    engine_version = run(args.llama_server, "--version")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "gpu": before["name"], "driver": before["driver"],
        "baseline_used_mib": before["used_mib"], "baseline_free_mib": before["free_mib"],
        "background_compute_process_count": len(before["processes"]),
        "runtime_sha256": digest(args.runtime),
        "node_version": run(args.node, "--version").stdout.strip(),
        "engine_version": (engine_version.stdout + engine_version.stderr).strip(),
        "engine_binary_sha256": digest(args.llama_server),
        "context_tokens": 2048, "parallel_slots": 1, "gpu_layers_requested": 99,
        "models": {
            "a": {"file": Path(args.model_a).name, "sha256": digest(args.model_a)},
            "b": {"file": Path(args.model_b).name, "sha256": digest(args.model_b)},
        },
        "steps": [],
        "scope": "Two text models on one shared physical GPU; real systemd user units and inference",
        "not_tested": ["ComfyUI/H3 media", "external residency leases", "sticky restoration", "multi-GPU scheduling"],
    }

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def install(unit, content):
        path = unit_dir / unit
        with path.open("x") as stream:
            stream.write(content)
        owned.append(path)

    def background_unchanged(snapshot, test_pids=()):
        return set(snapshot["processes"]) - set(test_pids) == set(before["processes"])

    manager_url = f"http://127.0.0.1:{args.manager_port}"
    worker_url = f"http://127.0.0.1:{args.text_port}"
    owner = "hardware-validation-" + uuid.uuid4().hex
    previous_pid = None
    previous_unit = None
    started = time.monotonic()
    try:
        for label, unit, model in zip(("a", "b"), WORKERS, (args.model_a, args.model_b)):
            install(unit, f"""[Unit]
Description=Temporary Fleet hardware validation worker {label}
[Service]
Type=simple
Environment=CUDA_VISIBLE_DEVICES={args.gpu_uuid}
ExecStart={args.llama_server} --model {model} --alias fleet-test-{label} --host 127.0.0.1 --port {args.text_port} --ctx-size 2048 --parallel 1 --gpu-layers 99 --threads 4 --no-webui --log-verbosity 4
Restart=no
KillMode=control-group
TimeoutStopSec=30
RuntimeMaxSec=600
StandardOutput=append:{work}/worker-{label}.log
StandardError=append:{work}/worker-{label}.log
""")
        env = {
            "MODEL_FLEET_HOST": "127.0.0.1", "MODEL_FLEET_PUBLIC_HOST": "127.0.0.1",
            "MODEL_FLEET_PORT": str(args.manager_port), "MODEL_FLEET_SYSTEMCTL_SCOPE": "user",
            "MODEL_FLEET_AVAILABLE_PROFILES": "qwen27-long,qwen35-long,idle",
            "MODEL_FLEET_GPU_UUID": args.gpu_uuid, "MODEL_FLEET_MEDIA_GPU_UUID": args.gpu_uuid,
            "MODEL_FLEET_ENCODER_GPU_UUID": "", "MODEL_FLEET_TEXT_HEALTH_URL": worker_url + "/health",
            "MODEL_FLEET_COMFY_HEALTH_URL": "http://127.0.0.1:1/system_stats",
            "MODEL_FLEET_CONTINUITY_HEALTH_URL": "http://127.0.0.1:1/health",
            "MODEL_FLEET_DATA_ROOT": str(work / "state"),
            "MODEL_FLEET_STATE_PATH": str(work / "state/state.json"),
            "MODEL_FLEET_OUTPUT_ROOT": str(work / "output"),
            "MODEL_FLEET_EXTERNAL_LEASE_STATE_PATH": str(work / "state/external-lease.json"),
            "MODEL_FLEET_MEDIA_CONTINUITY": "0", "MODEL_FLEET_MEDIA_PRELOAD": "0",
            "MODEL_FLEET_STICKY_MEDIA": "0", "MODEL_FLEET_EXTERNAL_LEASE_BASE_URL": "",
            "MODEL_FLEET_EXTERNAL_LEASE_ACQUIRE_URL": "", "MODEL_FLEET_EXTERNAL_LEASE_RENEW_URL": "",
            "MODEL_FLEET_EXTERNAL_LEASE_RELEASE_URL": "", "NODE_OPTIONS": "",
        }
        install(MANAGER, "[Unit]\nDescription=Temporary Fleet hardware validation manager\n[Service]\n"
                + "Type=simple\n" + "".join(f"Environment={k}={v}\n" for k, v in env.items())
                + f"ExecStart={args.node} {args.runtime}\nRestart=no\nRuntimeMaxSec=600\n"
                + f"StandardOutput=append:{work}/manager.log\nStandardError=append:{work}/manager.log\n")
        run("systemctl", "--user", "daemon-reload")
        run("systemctl", "--user", "start", MANAGER)
        for _ in range(100):
            try:
                if http(manager_url + "/health", timeout=2)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Manager did not become healthy")
        code, body = http(manager_url + "/lease/acquire", {"owner": owner, "ttl_ms": 600000})
        assert code == 200, body

        for number, label in enumerate(("a", "b", "a", "b"), 1):
            index = 0 if label == "a" else 1
            target = ("qwen27-long", "qwen35-long")[index]
            unit = WORKERS[index]
            print(json.dumps({"event": "transition_started", "step": number, "model": label}), flush=True)
            exit_observation, watcher = observe_exit(previous_pid) if previous_pid else ({}, None)
            t0 = time.monotonic()
            code, state = http(manager_url + "/transition", {
                "target": target, "owner": owner, "reason": "isolated real GPU validation",
            })
            elapsed = time.monotonic() - t0
            assert code == 200 and state["phase"] == "ready", state
            props = properties(unit)
            pid = int(props["MainPID"])
            snapshot = gpu_snapshot(args.gpu_uuid)
            assert pid in snapshot["processes"], "Worker has no CUDA allocation on the chosen GPU"
            assert snapshot["processes"][pid] >= 256, "Unexpectedly small GPU allocation"
            assert background_unchanged(snapshot, [pid]), "Background GPU process set changed"
            step = {
                "step": number, "model": label, "target_profile": target,
                "transition_to_ready_seconds": round(elapsed, 3),
                "worker_gpu_memory_mib": snapshot["processes"][pid],
                "total_gpu_used_mib": snapshot["used_mib"],
                "total_gpu_free_mib": snapshot["free_mib"],
                "background_processes_preserved": True,
            }
            if previous_pid:
                assert previous_pid not in snapshot["processes"]
                assert not Path(f"/proc/{previous_pid}").exists()
                watcher.join(timeout=1)
                old_exit = exit_observation.get("exit_observed_us", 0)
                new_start = int(props["ExecMainStartTimestampMonotonic"])
                assert old_exit > 0 and new_start >= old_exit, (old_exit, new_start)
                step.update(previous_worker_exited=True, previous_gpu_allocation_absent=True,
                            previous_exit_observed_before_new_start=True,
                            exit_observation_method="Linux pidfd + CLOCK_MONOTONIC",
                            previous_exit_observed_to_new_start_ms=round((new_start - old_exit) / 1000, 3))

            code, models = http(worker_url + "/v1/models")
            assert code == 200 and models["data"][0]["id"] == f"fleet-test-{label}", models
            t0 = time.monotonic()
            code, completion = http(worker_url + "/v1/chat/completions", {
                "model": f"fleet-test-{label}",
                "messages": [{"role": "user", "content": "What is 2 + 2? Reply with only the number."}],
                "temperature": 0, "seed": 42, "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            inference_seconds = time.monotonic() - t0
            assert code == 200, completion
            content = completion["choices"][0]["message"].get("content", "").strip()
            step.update(inference_seconds=round(inference_seconds, 3), response=content,
                        usage=completion.get("usage"), served_model=completion.get("model"))
            report["steps"].append(step)
            step["semantic_check_passed"] = bool(re.fullmatch(r"4[.!]?", content))
            save()
            code, rejected = http(manager_url + "/transition", {"target": "idle", "owner": "other-client"})
            assert code == 409, rejected
            assert int(properties(unit)["MainPID"]) == pid, "Rejected switch changed the worker"
            step["competing_owner_rejected_without_stopping_worker"] = True
            previous_pid, previous_unit = pid, unit
            save()
            print(json.dumps({"event": "step_completed", **step}), flush=True)

        t0 = time.monotonic()
        code, state = http(manager_url + "/transition", {"target": "idle", "owner": owner})
        assert code == 200 and state["phase"] == "ready", state
        after_idle = gpu_snapshot(args.gpu_uuid)
        assert background_unchanged(after_idle)
        assert abs(after_idle["used_mib"] - before["used_mib"]) <= 128
        assert not Path(f"/proc/{previous_pid}").exists()
        report["idle"] = {
            "transition_seconds": round(time.monotonic() - t0, 3),
            "gpu_used_mib": after_idle["used_mib"], "gpu_free_mib": after_idle["free_mib"],
            "delta_from_baseline_mib": after_idle["used_mib"] - before["used_mib"],
            "test_gpu_processes_absent": True, "background_processes_preserved": True,
        }
        assert http(manager_url + "/lease/release", {"owner": owner})[0] == 200
        report["status"] = "passed" if all(s["semantic_check_passed"] for s in report["steps"]) else "failed"
        if report["status"] == "failed":
            report["error"] = "One or more workers failed the semantic inference check; see per-step responses"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_errors = []
        # Stop only units whose files this invocation successfully created.
        for path in reversed(owned):
            result = run("systemctl", "--user", "stop", path.name, check=False)
            if result.returncode:
                cleanup_errors.append(path.name + ": stop failed")
        for label in ("a", "b"):
            log = work / f"worker-{label}.log"
            if log.exists():
                report["models"][label]["cuda_log_evidence"] = list(dict.fromkeys(
                    line.strip() for line in log.read_text(errors="replace").splitlines()
                    if re.search(r"offloaded \d+/\d+ layers|CUDA\d+.*(?:model|KV|compute) buffer", line)
                ))
        for path in owned:
            run("systemctl", "--user", "reset-failed", path.name, check=False)
            path.unlink(missing_ok=True)
        if owned:
            run("systemctl", "--user", "daemon-reload")
        cleanup_gpu = gpu_snapshot(args.gpu_uuid)
        report["cleanup"] = {
            "errors": cleanup_errors,
            "temporary_unit_files_removed": all(not p.exists() for p in owned),
            "background_processes_preserved": background_unchanged(cleanup_gpu),
            "gpu_used_mib": cleanup_gpu["used_mib"],
            "gpu_free_mib": cleanup_gpu["free_mib"],
            "delta_from_baseline_mib": cleanup_gpu["used_mib"] - before["used_mib"],
        }
        if cleanup_errors or not report["cleanup"]["background_processes_preserved"]:
            report["status"] = "failed"
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({"event": "finished", "status": report["status"], "cleanup": report["cleanup"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
