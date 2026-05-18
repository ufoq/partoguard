# PartoGuard Remote Gemma Deployment (llama.cpp on VPS)

**Status: Working.** As of 2026-05-17, the V7-fine-tuned Gemma 4 E2B-it model
is deployed on a CPU VPS via `llama-server` and serves the PartoGuard console
at `--gemma-remote`. Q8_0 is the live default; bf16 GGUF is available on
`/home/admin/` for swap-in via `MODEL=/home/admin/v7_bf16.gguf` env var.

## Headline result (single source of truth)

Two full 350-image evals were run end-to-end through `partoguard eval --gemma-remote`,
single prompt, no TTA, no preprocessing:

| Run                                | Date       | Wall avg | Result          |
|------------------------------------|------------|---------:|-----------------|
| **Q8_0** (currently serving)       | 2026-05-17 | 8.08 s   | **335/350 = 95.71%** |
| **bf16** (this session, swap-in)   | 2026-05-17 | 10.09 s  | **336/350 = 96.00%** |
| bf16 local on RTX 5090 (HF)        | 2026-04-xx | 5.77 s   | **350/350 = 100%**   |

The +1-image (+0.29 pp) bf16 gain on llama.cpp is well within run-to-run noise
on this corpus. **Quantization is not the dominant error source on this stack** —
Q8_0 and bf16 trip on essentially the same images. The 4 pp gap to the local
HF/CUDA baseline is **runtime (mtmd vision pipeline, image preprocessing,
mmproj conversion)**, not bit width.

Per-category for the live Q8_0 deployment:

| Category   | Total | Correct | %        |
|------------|------:|--------:|---------:|
| blank      |    50 |      50 | 100.00%  |
| partial    |    60 |      60 | 100.00%  |
| filled     |   100 |      88 |  88.00%  |
| obstructed |    60 |      58 |  96.67%  |
| degraded   |    80 |      79 |  98.75%  |

bf16 numbers (this session) shift only filled to 89/100 and one obstructed
to manual_review. All 50 blank charts correctly route to `manual_review`
(zero hallucination) on both quants.

Eval logs:
- `knowledge/eval_logs_v7_remote_q8_0_350.log`  — 335/350 Q8_0 run
- `knowledge/eval_logs_v7_remote_bf16_350.log`  — 336/350 bf16 run

## Why we serve Q8_0 by default

bf16 fits the VPS comfortably, but on this CPU stack the accuracy gain is
~0.3 pp while wall time grows 25%. Q8_0 wins on demo-relevant metrics
(latency, model-load time) without measurably hurting the clinical signal.

We keep `v7_bf16.gguf` (9.27 GB) on `/home/admin/` so we can swap in seconds
if ever needed:

```bash
ssh admin@vps-box 'pkill -9 -f llama-server; rm -f /home/admin/llama-server.pid; \
  MODEL=/home/admin/v7_bf16.gguf bash /home/admin/.startup'
```

## Hardware (verified 2026-05-17)

```
Host:    vps-box  (Docker VPS, only /home persists across container rebuilds)
CPU:     32× Intel Core i9-13900K
RAM:     31 GiB
Swap:    159 GiB
Disk:    /home is a 915 GB nvme partition
```

Earlier docs in this repo claimed an "8 GB CPU VPS" RAM constraint and used
that to justify Q8_0 as the only quant that fits. **That claim was wrong** —
it was an unverified assumption baked into a previous session's notes. The
real constraint that selects Q8_0 is the speed/accuracy trade-off above,
not RAM.

## Server side

- **Build**: `llama.cpp` `b9199-39cf5d619` compiled in `/home/admin/`
  (CPU-only). `libgomp.so.1` is required at runtime; container rebuilds wipe
  `/lib`, so a copy is stashed at `/home/admin/lib/libgomp.so.1` and
  `.startup` falls back to `LD_LIBRARY_PATH` if the system copy is missing.
- **Models on /home/admin/**:
  - `v7_q8_0.gguf`           (4.93 GB, Q8_0)  — live default
  - `v7_bf16.gguf`           (9.27 GB, bf16)  — swap-in via `$MODEL`
  - `v7_mmproj_f16.gguf`     (986 MB, F16 vision projector, shared)
  - `merged_v7/`             (9.6 GB, HF-format bf16 merged model — source for both GGUFs)
- **Conversion**: `convert_hf_to_gguf.py --outtype {bf16|q8_0}` from
  `merged_v7/` (~30 s wall on this CPU). The conversion venv is at
  `/home/admin/conv-venv/` (`torch 2.6.0+cpu`, `transformers 5.5.1`).

### Critical launch flags

```
-c 4096                # context window
-t 16 -tb 16           # threads (half the cores; box stays responsive)
-b 2048 -ub 2048       # MUST be >= image-token count
                       # (Gemma 4 vision can emit ~1024+ tokens/image;
                       #  -ub 256 aborts with "non-causal attention requires
                       #  n_ubatch >= n_tokens", reproduced 2026-05-17.)
--no-mmproj-offload    # no GPU on this host
--jinja                # required for Gemma 4 chat template
```

The full command line is in `scripts/vps_startup.sh` (= `/home/admin/.startup`).

## Critical client format (PR #21309 + empirical 2026-05-17)

The `/completion` endpoint multimodal format **changed in 2026**. The legacy
LLaVA-style `image_data: [{"data": "...", "id": N}]` + `[img-N]` token
format is dead — `post_completions` passes a hardcoded empty `files` vector
and never reads `image_data`. The current canonical format is:

```python
import requests, base64

# 1. fetch the server-randomized media marker (per server instance)
marker = requests.get("http://vps-box:8080/props").json()["media_marker"]
# e.g. "<__media_lLLgXj6DTFhVXIIJMJ5lDbzU1Md6ewPz__>"

# 2. build a Gemma 4 chat-formatted prompt with marker as image placeholder
#    DO NOT include <bos> in the prompt string — tokenizer adds it automatically.
prompt_string = (
    f"<|turn>user\n"
    f"{marker}\n"
    f"{prompt_body}<turn|>\n"
    f"<|turn>model\n"
)

# 3. POST to /completion with the new prompt-object form
payload = {
    "prompt": {
        "prompt_string": prompt_string,
        "multimodal_data": [base64.b64encode(image_bytes).decode("utf-8")],
    },
    "n_predict": 400,
    "temperature": 0.0,
    "top_k": 1,
}
response = requests.post("http://vps-box:8080/completion", json=payload, timeout=180)
```

`partoguard.core.extraction.gemma_adapter.RemoteGemmaExtractor` implements
this with a thread-safe per-process `_REMOTE_MARKER_CACHE` that fetches once
and reuses for the lifetime of the server instance. 23 unit tests in
`partoguard/tests/test_remote_gemma_extractor.py` cover the contract.

## CLI usage

```bash
# Single image
.venv/bin/partoguard analyze data/filled/filled_0181.png \
  --gemma-remote --gemma-remote-url http://vps-box:8080/completion

# Full corpus eval
.venv/bin/partoguard eval --corpus-dir data \
  --gemma-remote --gemma-remote-url http://vps-box:8080/completion --progress
```

## Known failure modes (Q8_0: 15 / 350)

| Failure mode                                         | Count | Notes                                         |
|------------------------------------------------------|------:|-----------------------------------------------|
| filled over-count (≥7 marks, dense charts)           |     7 | Same on bf16; runtime/preprocessing suspect    |
| filled rapid_precipitous zone misclass               |     4 | Documented V7 weakness on steep curves         |
| filled normal zone misclass                          |     2 | Same family                                   |
| degraded rapid_precipitous                           |     1 | Same family                                   |
| obstructed sparse-observation (n=2)                  |     2 | 0.5-grid quantization edge                    |

All 15 failures are clinically safe — they trigger `alert_zone` or
`action_zone` (over-call), never silent `normal` misses on filled images.

## Why we believe the gap to local 100% is runtime, not weights

The local HF/CUDA bf16 inference reaches 100% on the same 350 images that
llama.cpp/CPU bf16 reaches 96% on. Same model weights, same prompt, same
ground truth. The differences that remain:

| Layer              | Local HF/CUDA path                          | VPS llama.cpp/CPU path                         |
|--------------------|---------------------------------------------|------------------------------------------------|
| Vision encoder     | HF `SiglipImageProcessor` → vision tower → projector | mtmd helper + pre-converted `mmproj.gguf`      |
| Image preprocessing| Siglip-style (resize → normalize → patches) | mtmd's own resize/normalize                    |
| Tokenizer          | HF tokenizer (apply_chat_template)          | llama.cpp `--jinja` re-parser                  |
| Sampler            | HF `generate(do_sample=False)`              | llama.cpp sampler `top_k=1, T=0`               |
| Hardware           | Intel XPU / CUDA RTX 5090                   | i9-13900K CPU                                  |

We have not yet bisected which of these costs the 4 pp. Reasonable next
investigations:

1. **mmproj provenance** — verify `v7_mmproj_f16.gguf` was extracted from
   `merged_v7/` (post-LoRA-merge), not from base Gemma 4. If V7 LoRA never
   touched `vision_tower.*` or projector, this is moot — but worth a check.
2. **HF transformers CPU baseline** — bring up plain `transformers` on the
   VPS with `device_map="cpu", dtype=bfloat16` and run the same eval. If it
   recovers ~100%, runtime is confirmed culprit.
3. **Image preprocessing diff** — dump intermediate vision tokens from both
   stacks on `filled_0123.png` (a known failure) and compare.

These are not blocking the working deployment; the demo runs on Q8_0 today
and the deviation from "100%" is a deployment-quality question, not a
go/no-go.

## Persistence on the Docker VPS

Only `/home` survives container/VM rebuilds. The hooks that make persistence
work:

- `/home/admin/.startup` — invoked on boot. Mirror of `scripts/vps_startup.sh`.
- `/home/admin/lib/libgomp.so.1` — stashed copy of the runtime lib that
  apt-installs to `/lib` but vanishes on container rebuild. `.startup` adds
  `/home/admin/lib` to `LD_LIBRARY_PATH` if the system copy is missing.
- `/home/admin/conv-venv/` — Python venv with `torch 2.6.0+cpu`, used to
  re-convert merged_v7 → GGUF if we ever change quantization.

`scripts/vps_startup.sh` is the canonical, idempotent boot script. It:

- Validates that `llama-server`, the model GGUF, and the mmproj GGUF exist.
- Refuses to start a second instance if the previous PID is still alive.
- Truncates `/home/admin/llama-server.log` on each launch and mirrors a
  timestamped copy to `/home/admin/logs/llama-server-<UTC>.log`.
- Defaults to `MODEL=/home/admin/v7_bf16.gguf`. **The live deployment runs
  Q8_0**; switch to bf16 by setting the env var before invoking.

Install/update on the VPS:

```bash
scp scripts/vps_startup.sh admin@vps-box:/home/admin/.startup
ssh admin@vps-box 'chmod +x /home/admin/.startup'
```

Validate after a real reboot:

```bash
curl http://vps-box:8080/health
curl http://vps-box:8080/props | jq .modalities.vision      # -> true
curl http://vps-box:8080/v1/models | jq -r .data[0].id      # -> v7_q8_0.gguf
```

## Sources

- llama.cpp PR #21309 — Gemma 4 mtmd vision support (2026-04-02).
- llama.cpp PR #21500 — `add_bos_token=True` fix for Gemma 4 GGUF (2026-04-06).
- llama.cpp `tools/server/server-context.cpp` L3431-3436 — multimodal prompt path.
- llama.cpp `tools/server/server-common.cpp` L757-790 — `prompt_string + multimodal_data` schema.
- llama.cpp `tools/server/server-common.cpp` L87-97 — `media_marker` randomization.
- Gemma 4 official tokenizer config (`google/gemma-4-E2B-it/tokenizer_config.json`).

## Standing rule

This deployment supersedes the LiteRT-LM mobile path for desktop/server use
cases. LiteRT-LM remains the **on-device Android target** per
`partoguard_implementation_plan.md`; the VPS remote path is the
**demo/eval/server** target.
