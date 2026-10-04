# Validation record

Verified on 2026-10-04 with isolated local tests and a separate real GPU run.

## Real GPU result: passed for two text models

The unchanged legacy manager controlled real systemd user services on an NVIDIA
GeForce RTX 3090 (24 GiB), driver 580.142. It ran
**G9v3 3B → Ministral 3B → G9v3 3B → Ministral 3B → idle**. Every loaded model
served a real CUDA-backed generation request and correctly answered the simple
`2 + 2` check with `4`.

The engine was llama.cpp build 9780 (`1191758c5`), with Node 24.14.1, a 2,048-token
context and one inference slot. The temporary user-service identifiers were the
legacy `qwen27-long` and `qwen35-long` profiles; the actual models were the two
3B models below, not 27B/35B Qwen models. No mock services were used in this run.

| Activation | Time to Fleet ready | Worker VRAM | Generation check |
| --- | ---: | ---: | --- |
| G9v3 3B Q4_K_M | 5.681 s | 2,122 MiB | Passed |
| Ministral 3B Q8_0 | 6.892 s | 4,056 MiB | Passed |
| G9v3 3B restored | 5.683 s | 2,122 MiB | Passed |
| Ministral 3B restored | 7.709 s | 4,056 MiB | Passed |
| Idle | 0.953 s | No test worker remains | Baseline recovered |

Evidence checked during the passing run:

- Worker logs confirmed **53/53 G9v3 layers** and **27/27 Ministral layers**
  offloaded to the GPU, alongside real per-process GPU allocations.
- For every direct model swap, Linux pidfds observed the old process exiting
  before the new service's start timestamp. The observed gaps were 68.291,
  63.329, and 50.007 ms. The old process and its GPU allocation were absent
  after each swap.
- A competing owner received HTTP 409 at every activation, and the active
  worker kept the same PID.
- Total GPU usage returned exactly to the **4,153 MiB baseline**, with
  **19,972 MiB free** and **0 MiB residual test allocation**. The existing
  background compute process remained present throughout.
- Every temporary user-service file was removed and cleanup reported no errors.
- The manager's SHA-256 remained identical to the recorded legacy snapshot.

These timings include process startup, model loading, warmup, and the manager's
readiness checks. Model files were read for hashing before the measured
sequence, so filesystem caches were warm. They are observations for this
configuration, not cold-storage load benchmarks. The generation checks used
short answers; they are not a model-quality or sustained-throughput evaluation.

Full measurements, exact model/binary hashes, and CUDA log excerpts are in
[the passing JSON record](../validation/real-gpu-2026-10-04.json).

## Earlier attempts, kept separately

Earlier failures are excluded from the passing sequence and its timing table:

1. [Attempt 1](../validation/real-gpu-2026-10-04-attempt-1.json): a local
   `Qwen3-0.6B-Q8_0.gguf` artifact loaded onto the GPU but answered `2` to the
   generation check. The test stopped and cleanup recovered the baseline.
2. [Attempt 2](../validation/real-gpu-2026-10-04-attempt-2.json): the same artifact
   produced the same wrong answer with a different CUDA-enabled llama.cpp
   build. The cause of that artifact's inference failure was not established;
   it is not part of the validated pair.
3. [Attempt 3](../validation/real-gpu-2026-10-04-attempt-3.json): G9v3 passed
   inference and the next worker loaded, but the recorder could not read the
   exited service's timestamp after systemd discarded it. The recorder was
   corrected to use a Linux pidfd exit observation. The Fleet manager was not
   modified. Cleanup again recovered the baseline.

## Isolated test results

These separate checks ran locally on Node 20.19.2 and Python 3.13.5.

- Both JavaScript files passed `node --check`.
- All 6 integration tests passed in 28.145 seconds. The publication check
  repeated them successfully in 27.969 seconds, with both syntax checks passing.
- Both copied runtime files matched the deployed source SHA-256 hashes recorded
  in `PROVENANCE.md`.
- A publication pattern scan of all 18 included files found no private LAN
  IPv4 literals, deployed machine paths, real GPU UUIDs, personal email
  addresses, private-key headers, or common credential formats. Runtime
  references to environment variables and lease-token fields are code, not
  copied credential values. This is a scoped review, not a guarantee from an
  automated secret scanner.

| Test | Observed result with fake workers |
| --- | --- |
| Text → media → text → idle | Old worker stops before the new worker starts; idle leaves no managed worker active. |
| Concurrent transitions | A second transition waits while the first worker is not ready. |
| Lease ownership | Another owner cannot acquire/release the lease or change profile; rejected requests do not start workers. |
| Worker start failure | HTTP 500 and `phase: error`; the old worker remains stopped; a later transition can recover. |
| Lost worker | Status reports `phase: stale` after an active worker disappears. |
| MCP bridge | Initialization, tool listing, profile listing, and idle activation return successful responses. |

## Isolated test scope

The tests launch the actual copied Node manager and MCP bridge locally. They
replace `systemctl` and `nvidia-smi` with test executables and replace model
health endpoints with local HTTP servers. The manager receives a PATH containing
only those test executables and fresh temporary state directories. External
lease, preload, continuity, and sticky-media features are disabled.

This validates controller behavior without touching a real service, GPU, or
model. The subsequent hardware test above used separate temporary services and
did not replace the active deployment.

## Reproduce

```bash
node --check runtime/runtime.mjs
node --check runtime/runtime-mcp.mjs
python3 -m unittest discover -s tests -v
```

To reproduce the separate real GPU check, follow the explicit opt-in helper
instructions in [RECIPE.md](RECIPE.md). Normal unit-test discovery does not run
the hardware helper.

## Coverage limits

Hardware validation covers the two named text models on one shared RTX 3090.
ComfyUI/H3 generation, sticky restoration, external residency leases, different
GPU hardware, larger models, multi-GPU scheduling, long-running job admission,
and production MCP client compatibility still need their own validation.
