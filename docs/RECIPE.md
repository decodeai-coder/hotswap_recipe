# Using the legacy Fleet hot-swapping recipe

## 1. Set up the workers

Use a Linux machine with systemd user services, NVIDIA drivers, `nvidia-smi`, and
Node supporting the built-in `fetch` and `AbortSignal.timeout` APIs. Tests for
this package ran on Node 20.19.2; the test harness also needs Python 3.

First make each inference worker run successfully by itself. This package does
not install an inference engine, download weights, or configure a ComfyUI
workflow. The default example follows the hardware-tested pair of text-worker
profiles:

| Profile | Service | Readiness endpoint |
| --- | --- | --- |
| `qwen27-long` (model A) | `model-fleet-qwen27-long.service` | `http://127.0.0.1:8091/health` |
| `qwen35-long` (model B) | `model-fleet-qwen35-long.service` | `http://127.0.0.1:8091/health` |
| `idle` | Stops the manager's known workers | No worker endpoint |

Both text workers share one port because only one runs at a time. The measured
models were **G9v3 3B Q4_K_M** and **Ministral 3B Q8_0**, using llama.cpp build
9780 (`1191758c5`), 2,048-token context, one inference slot, four CPU threads,
and `--gpu-layers 99`. Exact filenames and hashes are recorded in
[the passing result](../validation/real-gpu-2026-10-04.json). Obtain suitable
weights and a CUDA-enabled engine separately; they are not bundled here.

`qwen27-long` and `qwen35-long` are legacy identifiers, not automatic model
discovery or claims that the tested models were Qwen models. This recipe keeps
those identifiers to preserve the snapshot. To add different identifiers, adapt `profileCatalog` in
the manager and `legacyProfiles`/tool configuration in the MCP bridge together.
The catalog's context, vision, and MTP fields are also historical metadata;
match them to the actual worker before using them for request admission.

Choose a GPU dedicated to this experiment and pin both worker services to its
UUID. The manager reports free VRAM but does not enforce a minimum free-memory
threshold or stop unrelated applications. Switching only works when the
selected model fits once the other managed worker exits.

## 2. Configure the user services

From the recipe directory, copy the example environment file and service files
into a temporary working directory and edit them before installation:

- Replace every `/absolute/path/...` and GPU placeholder.
- Set `MODEL_FLEET_NODE` to the absolute path reported by `command -v node`.
- Use absolute writable directories for state and output.
- Replace each worker's `ExecStart` with its independently verified launch
  command; the example model flags are starting points.
- Ensure none of these service names belong to another application. The legacy
  controller also knows `model-fleet-proxy.service` and
  `model-fleet-g9v3-media.service`, even when the minimal profile list is used.
- Keep `MEDIA_CONTINUITY`, `MEDIA_PRELOAD`, and `STICKY_MEDIA` disabled for this
  minimal setup. Their full environment variable names appear in the example.

Put the edited files in these locations:

```text
~/.config/model-fleet/runtime.env
~/.config/systemd/user/model-fleet-manager.service
~/.config/systemd/user/model-fleet-qwen27-long.service
~/.config/systemd/user/model-fleet-qwen35-long.service
```

Then load and start the manager, without enabling the workers at boot:

```bash
systemctl --user daemon-reload
systemctl --user start model-fleet-manager.service
curl --fail-with-body http://127.0.0.1:8199/health
curl --fail-with-body http://127.0.0.1:8199/profiles
curl --fail-with-body http://127.0.0.1:8199/status
```

The manager does not start a default model with sticky mode disabled. On a fresh
state directory, `phase: unknown` is expected until the first transition. Only
enable the manager at boot after the hardware validation below succeeds.

## 3. Request a profile

For a single-client demonstration, activate model A:

```bash
curl --fail-with-body --max-time 720 \
  -H 'Content-Type: application/json' \
  -d '{"target":"qwen27-long","reason":"model A task"}' \
  http://127.0.0.1:8199/transition
```

The manager stops conflicting known workers, starts the selected service, and
requires three consecutive successful readiness responses. Requests to change
profile are serialized. For the legacy defaults, text readiness can wait up to
10 minutes and media readiness up to 5 minutes; clients need suitable request
timeouts. A client timeout does not cancel a transition already running.

Call the inference worker separately after the manager returns `phase: ready`.
The `/transition` endpoint changes residency; it does not submit inference.

## 4. Protect a job and restore the previous profile

Applications sharing the manager should use a unique owner ID per job and take
a lease before any transition. The following single-job example uses
`example-job`; replace it with a unique identifier in an application.

```bash
curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"owner":"example-job","purpose":"model B job","ttl_ms":3600000}' \
  http://127.0.0.1:8199/lease/acquire

curl --fail-with-body --max-time 720 -H 'Content-Type: application/json' \
  -d '{"target":"qwen35-long","owner":"example-job","reason":"model B job"}' \
  http://127.0.0.1:8199/transition
```

Now submit inference directly to model B and wait for completion, for example:

```bash
curl --fail-with-body --max-time 120 -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2 + 2? Answer with only the number."}],"max_tokens":16,"temperature":0}' \
  http://127.0.0.1:8091/v1/chat/completions
```

Do not restore another profile while the current job is still running.

Once that job is terminal, restore the previously selected profile and then
release the lease, using the same owner ID:

```bash
curl --fail-with-body --max-time 720 -H 'Content-Type: application/json' \
  -d '{"target":"qwen27-long","owner":"example-job","reason":"restore after model B"}' \
  http://127.0.0.1:8199/transition

curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"owner":"example-job"}' \
  http://127.0.0.1:8199/lease/release
```

The application must handle failed HTTP responses instead of proceeding to
inference or cleanup blindly. A restore failure must be reported; the legacy
manager does not guarantee automatic rollback. To leave the managed workers
unloaded, transition to `idle` instead of restoring text.

In an application, choose the target from the task type, capture the previous
profile after acquiring the lease, perform the transition, run and await the
job, restore the desired profile, and release the lease. A client that crashes
after job submission must reconcile the job's status before unloading its
worker. Lease expiry does not prove the job has finished.

Leases are cooperative ownership, not authentication. Another owner receives
HTTP 409 while a lease is held. Leases are kept in memory, expire after their
configured TTL, and are lost on manager restart. TTL is clamped to 1 minute
through 6 hours. A long job must refresh the lease using the same owner before
expiry. Do not begin a competing job until the preceding job and any transition
have completed.

## 5. Optional MCP bridge

Start `runtime/runtime-mcp.mjs` with Node as a stdio MCP server, setting:

```text
MODEL_FLEET_URL=http://127.0.0.1:8199
MODEL_FLEET_AVAILABLE_PROFILES=qwen27-long,qwen35-long,idle
```

Its three tools are `get_runtime_status`, `list_runtime_profiles`, and
`activate_runtime_profile`. The legacy bridge does not expose lease acquisition
or release, and its activation schema does not advertise an owner field. Use it
for a single coordinated client, or extend it before using leased concurrent
jobs. The HTTP examples above cover the lease-aware path. This package tests the
bridge's basic request/response behavior, not broad client compatibility.

## 6. Optional legacy features

### Media-worker variant

The `model-fleet-comfy.service` template supports a text/media variant. Install
it only after independently validating your ComfyUI worker, add `media` to
`MODEL_FLEET_AVAILABLE_PROFILES` in both manager and MCP configuration, and use
`media` as the transition target. Its readiness URL is
`http://127.0.0.1:8190/system_stats`. Restart the idle manager to apply changed
configuration after all jobs have finished and all leases have been released.

Submit your workflow directly to ComfyUI and wait for that job to finish before
restoring text. With preload disabled, a healthy ComfyUI server does not mean
the workflow's weights are already loaded. The first job can still incur model
loading. This variant passed isolated controller tests; it has not had the
real GPU validation recorded for the two-text-model setup.

### Restoration and external leases

The unmodified source also contains an external residency-lease integration,
sticky media restoration, an optional continuity model, and an H3-specific
preload workflow. Those features depended on additional services or custom
ComfyUI nodes in the original deployment and are outside this minimal recipe.

Sticky media tries to return to media residency when no local lease is held. It
is not a job-aware general scheduler, and it can conflict with an unleased text
workflow. Keep it disabled until the external lease protocol and client
ownership rules have been adapted and tested.

## 7. Verify on a test GPU

After the isolated tests pass, run A → B → A → B → idle with a real generation
request after each activation. Record the following for every transition:

- The previous worker exits and releases its GPU allocation.
- The requested worker starts and becomes healthy.
- A real inference request produces a valid result.
- GPU UUID, free VRAM before/after, and elapsed load time.
- A competing owner is refused while the job's lease is active.
- `idle` stops all workers belonging to this recipe.

Use `nvidia-smi` and `journalctl --user -u <service>` for this evidence. The
isolated tests do not establish model quality, freed VRAM, load speed, memory
capacity, or real systemd service behavior. The separate
[hardware validation record](VALIDATION.md) covers one measured configuration.

### Optional hardware test helper

`tests/run_real_gpu_validation.py` is a real hardware test, excluded from the
normal `unittest` discovery command. Run it before installing the persistent
example units, or remove your stopped example units first: it deliberately
refuses existing units with the same names. It requires Linux/Python support for
`os.pidfd_open`, a working systemd user manager, a CUDA-enabled `llama-server`,
two suitable standalone GGUF models, and the copied Fleet manager. Its fixed
service names must be completely unused in that user's systemd scope. It
refuses existing units, occupied ports, models over 4 GiB each, or less than
8 GiB of free VRAM. These bounds are for the small validation fixture, not
general model-capacity guarantees.

After independently verifying your worker/model choices, replace every
placeholder and run from a user with access to the chosen GPU:

```bash
python3 tests/run_real_gpu_validation.py \
  --gpu-uuid GPU-REPLACE-WITH-YOUR-UUID \
  --model-a /absolute/path/model-a.gguf \
  --model-b /absolute/path/model-b.gguf \
  --llama-server /absolute/path/llama-server \
  --node /absolute/path/node \
  --runtime /absolute/path/fleet-hot-swap-recipe/runtime/runtime.mjs \
  --work-dir /absolute/path/fresh-validation-directory
```

The helper loads A → B → A → B, checks a simple generation result after each
load, verifies that a competing lease owner cannot unload the worker, and
transitions to `idle`. It observes process exit using Linux pidfds, checks real
GPU allocations, and writes `result.json` with model hashes and timings. Wrong
generation answers are recorded and make the run fail even if switching works.

The helper uses temporary ports 18199 (manager) and 18091 (worker), while the
persistent examples above use 8199 and 8091. It generates its own temporary
configuration with the same profile pair, GPU layer count, and context settings,
runs the unchanged
runtime, and does not read your persistent `runtime.env`.

The helper removes its own temporary units in cleanup and leaves its result and
worker logs in the selected work directory. Inspect those files before sharing
them; raw worker logs can include your model paths. All jobs must already be
finished before cleanup, and this helper's brief inference checks do not cover
long-running concurrent jobs. The example successful run and earlier failed
attempts are preserved in `validation/`.

## Operational limits of the original implementation

Keep the manager on loopback and under a trusted user. Its unauthenticated HTTP
API controls services; do not expose it directly on a public interface.

Transitions do not drain inference queues. Stopping a worker interrupts any
request still running on it. The caller must own job admission and completion.
There is no portable arbitrary-model routing, minimum-VRAM admission check,
multi-GPU scheduler, or guaranteed rollback. For a local worker start failure,
the manager records `phase: error` and the previous worker can remain stopped.

The status object's display URLs retain legacy ports; when changing health
URLs, use your configured worker URLs for inference. Only one manager should
control a given set of service names and GPU resources.
