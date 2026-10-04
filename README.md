# Model hot-swapping recipe

A recipe for sharing a GPU between models: stop the managed worker that
currently holds VRAM, start the requested worker, wait for its health endpoint,
and let the caller run its job. The included default configuration switches
between two text models; an optional media-worker example is also included.

This package preserves the legacy Fleet manager and its optional MCP bridge. The
two files in `runtime/` are unchanged source snapshots; the documentation,
configuration templates, and isolated tests were added for this recipe.

**Hardware tested:** the unchanged manager passed a real RTX 3090
test switching G9v3 3B and Ministral 3B through A → B → A → B → idle, with
generation checks after every load and full recovery of the test allocations.
This validates the two text-worker profiles; ComfyUI/H3, sticky restoration, and
external leases need separate hardware validation. See
[provenance](PROVENANCE.md) and [validation](docs/VALIDATION.md).

## How it works

```mermaid
sequenceDiagram
    participant C as Application or agent
    participant F as Fleet manager
    participant A as Model A worker
    participant B as Model B worker
    C->>F: Acquire lease; request model B profile
    F->>A: Stop model A service
    F->>B: Start model B service
    F->>B: Wait for three successful health checks
    F-->>C: Profile ready
    C->>B: Run job and wait for completion
    C->>F: Restore model A using the same lease
    F->>B: Stop model B service
    F->>A: Start model A service; wait for health
    C->>F: Release lease
```

The caller selects the profile for its task. The manager does not infer which
model a prompt needs, route inference requests, or discover arbitrary GPU
processes. "Hot-swapping" here means restarting model services, so model loading
still adds latency and worker sessions are not preserved.

## Included

| Path | Purpose |
| --- | --- |
| `runtime/runtime.mjs` | Legacy HTTP manager: serialized transitions, leases, readiness, status, optional media restoration. |
| `runtime/runtime-mcp.mjs` | Optional stdio MCP bridge exposing status, profiles, and profile activation. |
| `examples/runtime.env.example` | Configuration for two text workers and idle, matching the tested profile pair. |
| `examples/systemd/` | User-service templates; worker executable and model paths require editing. |
| `docs/RECIPE.md` | Setup, requests, integration rules, and limits of the legacy implementation. |
| `tests/test_runtime.py` | Isolated integration tests using fake services and health endpoints. |
| `tests/run_real_gpu_validation.py` | Explicit opt-in hardware test using temporary user services and real model inference. |
| `validation/` | Sanitized measurements from the passing hardware run and the separately retained failed attempts. |

Model weights, inference engines, ComfyUI, custom nodes, vendor dependencies,
credentials, runtime state, and generated media are not included.

## Start here

1. Read the [recipe](docs/RECIPE.md), especially its worker ownership and job
   completion rules.
2. Run the isolated tests from this directory:

   ```bash
   node --check runtime/runtime.mjs
   node --check runtime/runtime-mcp.mjs
   python3 -m unittest discover -s tests -v
   ```

3. Configure two independently working text models using the example user
   services. The tested pair used G9v3 3B Q4_K_M and Ministral 3B Q8_0 with
   llama.cpp, a 2,048-token context, and one inference slot. Exact artifact
   hashes and the separate hardware-test command are linked in the recipe.
4. Run the opt-in hardware helper on a suitable test GPU before adapting the
   configuration to other models or a media workflow.

The manager and bridge use Node built-ins and need no npm install. The local
test versions and the limits of that verification are recorded in
[VALIDATION.md](docs/VALIDATION.md).

## Privacy and licensing

Examples use placeholder paths and GPU selectors. Published test records omit
host addresses, GPU UUIDs, process IDs, private paths, and raw logs. Review your
own configuration and logs before sharing results.

No license was supplied with the recovered runtime snapshot, and this package
does not assign it a new license. See [PROVENANCE.md](PROVENANCE.md). Public
visibility alone does not grant general reuse permissions; see
[GitHub's licensing guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
