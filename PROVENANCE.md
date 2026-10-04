# Source provenance

The following files were copied unchanged from a deployed legacy Fleet runtime
on 2026-10-04. The inspected deployment no longer uses this legacy manager as its
active media controller. Its presence on disk is not evidence of a currently
working GPU deployment.

The copied manager was subsequently tested unchanged in a temporary hardware
validation setup. That test used real systemd user services, CUDA inference,
and two small text models. It did not replace the host's active media controller.
The temporary units were removed after each attempt. The recorded source hashes
still match both runtime files.

| File | SHA-256 |
| --- | --- |
| `runtime/runtime.mjs` | `b430647149a404ed7d667d7599234fffb92ef99122cc36e6a22c6adbba93122c` |
| `runtime/runtime-mcp.mjs` | `b6f471771b9b1e7ec026db2a46bb69158d27548766d88dfe2c5005efc81de014` |

The source snapshot did not include a top-level license or a Git commit ID.
This publication preserves that provenance limitation and does not assign a
new license to the recovered source. An upstream source-repository link and
reuse license remain unspecified. Public availability should not be read as a
general license grant.

Documentation, tests, and example configurations were prepared separately for
this package. The worker service examples are templates, not copies of a live
deployment's configuration. They require real executable paths and a GPU UUID.

No third-party implementations or model weights were copied. Inference engines,
models, and any optional ComfyUI extensions must be obtained separately under
their own terms. No support for those dependencies is implied by copying the
manager.
