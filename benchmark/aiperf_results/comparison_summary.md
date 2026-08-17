# IORails vs LLMRails Benchmark — 8×H100 Results

## Parser Bug Note (affects Runs 1, 2, and Scenario Sweeps v1)

All runs prior to 2026-07-23 used `nemotron-3.5-content-safety 2.0.5-variant`, which returns
verdicts in a `User Safety: safe/unsafe` text format. The output parser referenced in
`examples/configs/iorails_vs_llmrails_benchmark/prompts.yml`
(`nemotron_reasoning_parse_prompt_safety`) expected a `Prompt harm: harmful/unharmful` format and
failed to match, defaulting to `is_safe=False` on every request. Result: **100% of requests were
blocked at the input rail** — the main LLM NIM was never invoked. Throughput/latency numbers from
those runs reflect how fast each engine can process and reject a request through the input rail
only, not end-to-end guardrailed inference.

Fix: two new parsers (`nemotron_35_parse_prompt_safety` / `nemotron_35_parse_response_safety`) were
added to `nemoguardrails/llm/output_parsers.py` and registered in `nemoguardrails/llm/taskmanager.py`
(commit 7c4c59c73, NGUARD-872). The canonical scenario sweep results are in **Scenario Sweeps v2** below.

---

## Run 1 (2026-07-17, 60s per level) ⚠ parser bug — input-blocked only
**Hardware:** 8× H100 80GB (Brev node brev-ymkkdj0q4)
**NIMs:** Nemotron 3 Nano 2.0.8 (port 8000) + Nemotron 3.5 Content Safety 2.0.5-variant (port 8001)
**Guardrails:** branch dev/schilton/iorails-vs-llmrails-benchmarking
**Sweep:** c=1,2,4,8,16,32,64,128,256 × 60s each, aiperf profile mode

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  9.72 |             100.7 |             103.6 |               12.45 |             78.8 |             81.4 |
|           2 |                 15.27 |             128.9 |             142.3 |               22.75 |             86.6 |             91.8 |
|           4 |                 23.10 |             168.8 |             313.9 |               43.50 |             90.6 |             94.6 |
|           8 |                 24.31 |             313.1 |             583.8 |               80.82 |             94.7 |            137.9 |
|          16 |                 27.58 |             580.2 |             914.9 |              118.20 |            126.7 |            314.3 |
|          32 |                 27.56 |           1,149.7 |           2,548.7 |              150.14 |            194.6 |            413.1 |
|          64 |                 29.83 |           1,669.5 |           7,729.5 |              152.44 |            403.3 |            711.5 |
|         128 |                 28.77 |           3,455.7 |          15,593.5 |              100.52 |          1,020.9 |          5,045.7 |
|         256 |                 23.25 |           6,859.1 |          33,715.0 |              154.43 |          1,398.8 |         10,382.4 |

**Note:** IORails c=128 anomalous throughput dip (100 req/s vs ~152 at c=64 and c=256) — likely a scheduling artifact. Resolved in Run 2.

---

## Run 2 (2026-07-20–21, 120s per level) ⚠ parser bug — input-blocked only
**Hardware:** 8× H100 80GB (Brev node brev-f7a12xlhw)
**NIMs:** Nemotron 3 Nano 2.0.8 (port 8000) + Nemotron 3.5 Content Safety 2.0.5-variant (port 8001)
**Guardrails:** branch dev/schilton/iorails-vs-llmrails-benchmarking (tip a662a7970)
**Sweep:** c=1,2,4,8,16,32,64,128,256 × 120s each, aiperf profile mode

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                 11.43 |              86.5 |              90.8 |               16.96 |             58.2 |             61.1 |
|           2 |                 16.08 |             122.6 |             140.9 |               30.80 |             63.5 |             68.2 |
|           4 |                 24.41 |             162.9 |             250.3 |               57.60 |             67.3 |             73.3 |
|           8 |                 26.38 |             289.4 |             699.7 |              103.58 |             75.8 |             83.6 |
|          16 |                 28.25 |             556.7 |           1,111.9 |              166.83 |             91.1 |            170.9 |
|          32 |                 29.57 |             813.5 |           4,260.5 |              188.33 |            164.5 |            256.0 |
|          64 |                 29.89 |           1,617.1 |           8,545.0 |              185.88 |            344.9 |            481.5 |
|         128 |                 28.67 |           3,391.5 |          16,450.6 |              188.63 |            685.5 |            869.1 |
|         256 |                 26.17 |           6,874.7 |          33,085.2 |              189.72 |          1,396.8 |          1,961.6 |

---

## Scenario Sweeps v1 (2026-07-21, 120s per level) ⚠ INVALID — parser bug, 100% input-blocked

Results omitted. The ~10 req/s throughput and ~85ms P50 reflect CS NIM input-check latency only;
the main LLM NIM was never invoked. See Scenario Sweeps v2 for valid end-to-end results.

---

## Scenario Sweeps v2 (2026-07-22–23, 300s per level) ⚠ SUPERSEDED — max_tokens not forwarded, unconstrained LLM output
**Hardware:** 8× H100 80GB (Brev node brev-sax6j9j37)
**NIMs:** Nemotron 3 Nano 2.0.8 (port 8000) + Nemotron 3.5 Content Safety 2.0.5-variant (port 8001)
**Guardrails:** branch dev/schilton/iorails-vs-llmrails-benchmarking (tip 7c4c59c73)
**Scenarios:** defined in Tim Gasser's [WIP] IORails Tech Blog Plan
**Configs:** `benchmark/aiperf/configs/scenario_*_sweep.yaml`

Response classification by output sequence length (OSL):
- **Safe** (OSL > 20): request passed through input rail, main LLM invoked, output rail passed
- **Blocked** (OSL 6–11): input or output rail blocked the request (Guardrails refusal message)
- **Overload** (OSL 1–5): IORails fast-fail response under saturation; LLMRails queues instead

---

### Dialog (input 100 tokens / output 100 tokens / std 5)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  0.20 |           5,245.0 |           7,761.1 |                0.20 |          4,899.5 |          8,908.0 |
|           2 |                  0.36 |           5,558.1 |           8,789.3 |                0.36 |          5,442.4 |         13,005.1 |
|           4 |                  0.58 |           6,819.4 |          12,471.2 |                0.67 |          5,919.0 |         13,757.4 |
|           8 |                  0.94 |           8,563.3 |          14,293.1 |                1.06 |          7,488.1 |         13,703.3 |
|          16 |                  1.52 |          10,529.0 |          19,261.2 |                1.63 |          9,597.2 |         24,609.4 |
|          32 |                  2.37 |          13,312.9 |          24,243.0 |                2.43 |         12,641.8 |         30,163.9 |
|          64 |                  3.57 |          17,688.5 |          31,455.2 |                3.95 |         16,075.3 |         30,385.3 |
|         128 |                  5.34 |          23,175.4 |          41,877.5 |                5.31 |         23,916.3 |         35,971.1 |
|         256 |                  6.70 |          35,278.3 |          65,759.5 |               12.29 |         15,843.9 |         40,988.3 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |          98.3% |            1.7% |           0.0% |         98.3% |           1.7% |          0.0% |
|           2 |          98.2% |            1.8% |           0.0% |         98.1% |           1.9% |          0.0% |
|           4 |          97.7% |            2.3% |           0.0% |         98.0% |           2.0% |          0.0% |
|           8 |          97.9% |            2.1% |           0.0% |         97.8% |           1.9% |          0.3% |
|          16 |          98.0% |            2.0% |           0.0% |         97.3% |           2.0% |          0.6% |
|          32 |          98.0% |            2.0% |           0.0% |         97.0% |           2.1% |          1.0% |
|          64 |          97.8% |            2.2% |           0.0% |         96.3% |           2.1% |          1.6% |
|         128 |          97.9% |            2.1% |           0.0% |         85.1% |           2.1% |         12.8% |
|         256 |          97.7% |            2.2% |           0.1% |         37.7% |           2.1% |         60.2% |

---

### Code Generation (input 200 tokens / output 2000 tokens / std 10)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  0.15 |           6,620.7 |          11,789.8 |                0.18 |          5,787.8 |          8,401.3 |
|           2 |                  0.25 |           7,824.5 |          13,085.2 |                0.32 |          6,258.5 |          9,784.6 |
|           4 |                  0.52 |           8,011.9 |          14,723.5 |                0.56 |          7,172.2 |         13,020.8 |
|           8 |                  0.87 |           9,275.1 |          16,013.4 |                0.94 |          8,395.6 |         17,476.2 |
|          16 |                  1.40 |          11,516.5 |          20,547.5 |                1.47 |         10,617.7 |         21,540.9 |
|          32 |                  2.16 |          14,639.7 |          27,088.0 |                2.33 |         13,542.2 |         25,121.8 |
|          64 |                  3.19 |          19,735.2 |          35,271.5 |                3.51 |         17,977.4 |         30,377.8 |
|         128 |                  4.90 |          25,391.2 |          45,537.5 |                4.85 |         26,635.8 |         40,501.4 |
|         256 |                  6.31 |          38,466.6 |          69,434.6 |               12.04 |         15,775.0 |         41,459.8 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |          97.8% |            2.2% |           0.0% |         98.1% |           1.9% |          0.0% |
|           2 |          97.3% |            2.7% |           0.0% |         97.9% |           2.1% |          0.0% |
|           4 |          97.4% |            2.6% |           0.0% |         97.0% |           3.0% |          0.0% |
|           8 |          97.3% |            2.7% |           0.0% |         97.2% |           2.8% |          0.0% |
|          16 |          97.1% |            2.9% |           0.0% |         96.6% |           2.9% |          0.5% |
|          32 |          96.9% |            3.1% |           0.0% |         96.9% |           3.0% |          0.1% |
|          64 |          96.9% |            3.1% |           0.0% |         95.1% |           3.1% |          1.8% |
|         128 |          96.7% |            3.3% |           0.0% |         73.6% |           3.2% |         23.2% |
|         256 |          95.9% |            3.9% |           0.3% |         28.3% |           3.2% |         68.5% |

---

### RAG (input 4000 tokens / output 200 tokens / std 10)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  0.79 |             791.9 |           9,086.7 |                0.24 |          4,733.3 |         17,187.8 |
|           2 |                  1.44 |             900.1 |           9,363.9 |                0.50 |          3,078.8 |         10,485.0 |
|           4 |                  2.43 |             979.1 |          11,385.2 |                0.79 |          4,724.8 |         16,795.2 |
|           8 |                  3.93 |           1,275.7 |          13,630.5 |                1.26 |          6,157.4 |         21,601.0 |
|          16 |                  5.72 |           1,805.5 |          19,664.5 |                1.95 |          8,004.8 |         26,930.1 |
|          32 |                  7.29 |           2,792.5 |          29,363.9 |                2.94 |         10,787.9 |         30,851.1 |
|          64 |                  8.07 |           6,743.5 |          29,892.8 |                4.49 |         13,958.6 |         31,001.9 |
|         128 |                  7.83 |          15,485.3 |          45,613.6 |                7.09 |         19,706.0 |         40,986.2 |
|         256 |                  7.55 |          37,123.0 |          60,548.7 |               17.95 |         15,685.5 |         41,609.2 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |          58.7% |           41.3% |           0.0% |         45.7% |          54.3% |          0.0% |
|           2 |          61.0% |           39.0% |           0.0% |         48.0% |          52.0% |          0.0% |
|           4 |          60.3% |           39.7% |           0.0% |         44.5% |          55.1% |          0.4% |
|           8 |          61.2% |           38.8% |           0.0% |         46.9% |          53.1% |          0.0% |
|          16 |          60.6% |           39.4% |           0.0% |         47.5% |          52.0% |          0.5% |
|          32 |          60.6% |           39.4% |           0.0% |         43.2% |          53.7% |          3.1% |
|          64 |          60.6% |           39.4% |           0.0% |         39.0% |          49.0% |         12.0% |
|         128 |          59.3% |           40.2% |           0.5% |         18.4% |          45.2% |         36.4% |
|         256 |          58.6% |           41.3% |           0.1% |          7.0% |          39.6% |         53.4% |

**Note:** RAG and Agent block rates differ between engines (~40% LLMRails vs ~54% IORails at c=1) because aiperf regenerates input corpora independently per run; different Shakespeare passages are sampled, affecting the CS NIM verdict distribution. Cross-engine block rate comparisons for large-context scenarios are not apples-to-apples.

---

### Agent (input 8000 tokens / output 4000 tokens / std 100)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  0.65 |           1,040.2 |           6,844.2 |                0.24 |          4,205.0 |         15,042.0 |
|           2 |                  1.13 |           1,129.0 |          11,280.4 |                0.41 |          5,158.5 |         15,664.5 |
|           4 |                  1.65 |           1,460.5 |          14,330.2 |                0.66 |          6,489.6 |         25,487.7 |
|           8 |                  2.58 |           1,936.2 |          18,629.2 |                1.06 |          7,665.8 |         26,360.1 |
|          16 |                  3.66 |           3,452.8 |          20,038.1 |                1.63 |         10,329.1 |         30,339.2 |
|          32 |                  4.41 |           6,622.2 |          23,829.2 |                2.44 |         13,900.0 |         30,927.2 |
|          64 |                  5.16 |          11,535.0 |          33,180.2 |                3.66 |         18,777.6 |         32,771.7 |
|         128 |                  5.31 |          23,495.4 |          48,379.1 |                5.62 |         29,573.8 |         41,627.0 |
|         256 |                  4.92 |          49,933.1 |          82,310.1 |               10.03 |         20,890.0 |         48,988.4 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |          69.6% |           30.4% |           0.0% |         47.9% |          52.1% |          0.0% |
|           2 |          67.8% |           32.2% |           0.0% |         48.0% |          52.0% |          0.0% |
|           4 |          70.7% |           29.1% |           0.2% |         51.3% |          48.2% |          0.5% |
|           8 |          70.0% |           30.0% |           0.0% |         48.4% |          50.9% |          0.6% |
|          16 |          70.6% |           29.4% |           0.0% |         49.0% |          49.8% |          1.2% |
|          32 |          71.1% |           28.9% |           0.0% |         45.5% |          50.5% |          4.0% |
|          64 |          70.6% |           29.3% |           0.1% |         40.1% |          42.4% |         17.4% |
|         128 |          69.5% |           29.4% |           1.1% |         18.1% |          35.2% |         46.7% |
|         256 |          69.4% |           30.5% |           0.1% |          9.9% |          29.6% |         60.5% |

---

## Baseline LLM Sweep (2026-07-23, 300s per level) — direct LLM, no Guardrails
**Hardware:** 8× H100 80GB (Brev node brev-sax6j9j37)
**NIM:** Nemotron 3 Nano 2.0.8 (port 8000) — no content safety NIM in the request path
**Configs:** `benchmark/aiperf/configs/scenario_*_sweep.yaml` (url=http://localhost:8000, health_check_endpoint=/v1/models)
**Purpose:** Establish raw LLM throughput ceiling per scenario to quantify Guardrails engine overhead

| Scenario | Concurrency | tput (req/s) | P50 (ms) | P99 (ms) |
|----------|-------------|--------------|----------|----------|
| Dialog (100 in / 100 out) | 1 | 3.48 | 286.5 | 312.7 |
| | 2 | 5.95 | 336.0 | 365.3 |
| | 4 | 9.51 | 420.1 | 471.0 |
| | 8 | 14.25 | 561.1 | 619.2 |
| | 16 | 20.23 | 792.0 | 871.9 |
| | 32 | 28.89 | 1,110.2 | 1,228.6 |
| | 64 | 40.59 | 1,576.8 | 1,767.3 |
| | 128 | 54.81 | 2,337.0 | 2,635.1 |
| | 256 | 63.95 | 4,123.3 | 4,794.1 |
| RAG (4000 in / 200 out) | 1 | 1.65 | 605.2 | 649.8 |
| | 2 | 2.79 | 714.3 | 806.4 |
| | 4 | 4.32 | 927.9 | 1,055.0 |
| | 8 | 6.11 | 1,310.3 | 1,446.5 |
| | 16 | 8.20 | 1,952.7 | 2,192.5 |
| | 32 | 10.54 | 3,030.2 | 3,415.9 |
| | 64 | 13.35 | 4,783.8 | 5,461.9 |
| | 128 | 16.71 | 7,577.4 | 9,533.9 |
| | 256 | 19.31 | 12,871.5 | 19,486.1 |
| Code Gen (200 in / 2000 out) | 1 | 0.20 | 5,256.6 | 5,319.7 |
| | 2 | 0.39 | 5,686.4 | 5,824.6 |
| | 4 | 0.66 | 6,662.8 | 6,774.2 |
| | 8 | 1.08 | 8,184.9 | 8,335.8 |
| | 16 | 1.66 | 10,517.1 | 10,756.6 |
| | 32 | 2.55 | 13,673.7 | 13,951.2 |
| | 64 | 3.94 | 17,627.3 | 17,996.3 |
| | 128 | 5.95 | 23,164.8 | 23,885.6 |
| | 256 | 8.32 | 32,128.7 | 33,541.6 |
| Agent (8000 in / 4000 out) | 1 | 0.20 | 5,007.3 | 10,448.7 |
| | 2 | 0.31 | 6,428.9 | 12,120.0 |
| | 4 | 0.53 | 7,563.0 | 14,182.8 |
| | 8 | 0.88 | 9,183.8 | 17,742.2 |
| | 16 | 1.35 | 12,016.0 | 23,581.7 |
| | 32 | 1.97 | 16,290.1 | 32,815.1 |
| | 64 | 2.74 | 22,586.3 | 45,661.7 |
| | 128 | 3.68 | 32,845.2 | 66,602.4 |
| | 256 | 4.35 | 52,423.7 | 112,254.6 |

**Key finding:** Throughput continues scaling through c=256 without saturating for any scenario —
the LLM NIM is the bottleneck and data parallelism will be needed before Guardrails engine
overhead becomes visible in end-to-end measurements.

**Note on Guardrails vs. baseline comparison:** The Scenario Sweeps v2 Guardrails latency
(e.g. dialog P50=5,245ms at c=1) is far higher than the baseline (287ms) — the gap exceeds
what two CS NIM calls (~180ms total) can explain. Per-request OSL data from the Guardrails sweep
shows safe dialog responses reaching 2,982 tokens despite the aiperf config requesting 100,
suggesting `max_tokens` is not being forwarded from the client request to the main LLM. The
Guardrails vs. baseline comparison is therefore not apples-to-apples; the LLMRails vs. IORails
comparison remains valid since both engines share the same issue.

---

## Key Findings (Scenario Sweeps v2)

### Safety classification (consistent across concurrency levels)
- **Dialog**: ~98% safe, ~2% blocked — CS NIM flags ~1 in 50 short Shakespeare passages
- **Code Gen**: ~97% safe, ~3% blocked — similar to dialog
- **RAG** (LLMRails): ~60% safe, ~40% blocked — long passages trigger more CS NIM flags
- **Agent** (LLMRails): ~70% safe, ~30% blocked
- Block rates are stable across all concurrency levels for LLMRails, confirming deterministic CS NIM behavior under load

### Throughput comparison (safe + blocked; excludes overload)
- At **c=1 through c=64**, IORails and LLMRails throughput is within ~10% for dialog and code gen — LLM inference time dominates and engine overhead is proportionally small
- At **c=256**, IORails total throughput is 1.8–2.4× higher than LLMRails, driven primarily by overload fast-failing rather than genuine inference acceleration
- RAG and Agent: LLMRails has higher effective throughput at most concurrency levels for these scenarios, because blocked requests (40–50% of total) return quickly and inflate the request rate without LLM cost

### IORails overload behavior
- IORails begins returning fast-fail overload responses (OSL 1–5 tokens) at c=128+ for dialog/code gen; at c=32+ for RAG/agent
- At c=256, 60–70% of IORails dialog/code gen responses are overload fast-fails; LLMRails overload rate is <1% at all concurrency levels
- LLMRails has no work queue — it keeps accepting requests until it falls over. IORails uses a bounded work queue and rejects new requests when the queue is full; the fast-fail responses are intentional overload protection, not a failure mode

### LLM inference dominates latency
- Run 1/2 (input-blocked only) showed IORails 5–6× faster than LLMRails. With real end-to-end inference, the advantage at moderate concurrency (c=1–64) narrows to ~5–15% — the Guardrails engine overhead is small relative to LLM inference time
- The IORails advantage re-emerges at c=128+ through overload handling, which reduces queue depth and P50/P99 latency even when many of those fast responses are non-productive

---

## Scenario Sweeps v3 (2026-07-29, 300s per level) ← canonical
**Hardware:** 8× H100 80GB (Brev node brev-sax6j9j37)
**NIMs:** Nemotron 3 Nano 2.0.8 (port 8000) + Nemotron 3.5 Content Safety 2.0.5-variant (port 8001)
**Guardrails:** branch dev/schilton/iorails-vs-llmrails-benchmarking (tip d94c150ba)
**Fix vs v2:** `use_legacy_max_tokens: true` — aiperf now sends `max_tokens` instead of `max_completion_tokens`; Guardrails forwards it to the main LLM. Validated: dialog OSL mean 100.7 tokens (was ~2,982 in v2). NGUARD-873 workaround.

Response classification by output sequence length (OSL):
- **Safe** (OSL > 20): request passed through input rail, main LLM invoked, output rail passed
- **Blocked** (OSL 6–20): input or output rail blocked the request (Guardrails refusal message)
- **Overload** (OSL 1–5): IORails fast-fail response under saturation; LLMRails queues instead

---

### Dialog (input 100 tokens / output 100 tokens / std 5)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  2.16 |             461.6 |             488.4 |                2.45 |            406.9 |            431.8 |
|           2 |                  3.73 |             536.1 |             589.8 |                4.33 |            461.0 |            492.2 |
|           4 |                  6.02 |             653.3 |             900.1 |                7.54 |            530.5 |            572.0 |
|           8 |                  8.74 |             918.4 |           1,307.2 |               12.33 |            648.0 |            704.9 |
|          16 |                 11.20 |           1,438.5 |           2,197.1 |               19.38 |            825.1 |            921.0 |
|          32 |                 12.94 |           2,402.8 |           4,157.4 |               29.47 |          1,084.2 |          1,225.1 |
|          64 |                 14.50 |           4,083.0 |           9,840.9 |               43.77 |          1,466.9 |          1,694.7 |
|         128 |                 14.20 |           8,176.4 |          22,214.6 |               47.81 |          2,669.5 |          3,128.9 |
|         256 |                 13.59 |          17,648.1 |          44,964.6 |               47.40 |          5,429.9 |          5,916.1 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|           2 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|           4 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|           8 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|          16 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|          32 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|          64 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|         128 |          99.9% |            0.1% |           0.0% |        100.0% |           0.0% |          0.0% |
|         256 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |

**IORails peak:** 47.8 req/s vs LLMRails 14.5 req/s → **3.3× throughput**. IORails P99 at c=256: 5,916ms vs 44,965ms → **7.6× lower**. No overload fast-fails observed.

---

### RAG (input 4,000 tokens / output 200 tokens / std 10)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  1.16 |             886.6 |             990.3 |                1.27 |            790.8 |            838.4 |
|           2 |                  2.15 |             933.0 |           1,151.2 |                2.32 |            865.1 |            924.5 |
|           4 |                  3.43 |           1,150.4 |           2,373.5 |                3.89 |          1,033.5 |          1,124.0 |
|           8 |                  4.78 |           1,636.3 |           3,157.1 |                5.99 |          1,339.0 |          1,491.2 |
|          16 |                  5.48 |           2,779.1 |           5,059.3 |                8.44 |          1,902.9 |          2,231.0 |
|          32 |                  5.44 |           5,420.5 |          12,178.5 |               11.08 |          2,883.6 |          3,568.7 |
|          64 |                  5.36 |          10,671.8 |          27,882.4 |               14.05 |          4,537.8 |          5,643.2 |
|         128 |                  5.16 |          22,021.5 |          54,540.5 |               15.16 |          8,344.5 |          9,785.0 |
|         256 |                  5.48 |          47,151.6 |          57,408.2 |               14.94 |         12,065.6 |         22,937.3 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |          83.9% |           16.1% |           0.0% |         95.3% |           4.7% |          0.0% |
|           2 |          84.3% |           15.7% |           0.0% |         95.1% |           4.9% |          0.0% |
|           4 |          85.1% |           14.9% |           0.0% |         95.0% |           5.0% |          0.0% |
|           8 |          85.1% |           14.9% |           0.0% |         95.1% |           4.9% |          0.0% |
|          16 |          85.6% |           14.4% |           0.0% |         94.7% |           5.3% |          0.0% |
|          32 |          85.3% |           14.6% |           0.1% |         94.9% |           5.1% |          0.0% |
|          64 |          83.2% |           16.8% |           0.0% |         95.2% |           4.8% |          0.0% |
|         128 |          84.8% |           15.2% |           0.0% |         94.9% |           5.1% |          0.0% |
|         256 |          84.3% |           15.6% |           0.1% |         94.9% |           5.1% |          0.0% |

**IORails peak:** 15.2 req/s vs LLMRails 5.5 req/s → **2.7× throughput**. IORails P99 at c=256: 22,937ms vs 57,408ms → **2.5× lower**. Block rate difference (~15% LLMRails vs ~5% IORails) reflects different aiperf input corpora between runs — cross-engine block rate comparisons are not reliable.

---

### Code Generation (input 200 tokens / output 2,000 tokens / std 10)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  0.19 |           5,495.2 |           5,547.3 |                0.20 |          5,384.9 |          5,437.0 |
|           2 |                  0.34 |           6,016.5 |           6,081.9 |                0.37 |          5,904.9 |          5,986.2 |
|           4 |                  0.59 |           7,027.5 |           7,242.3 |                0.63 |          6,891.6 |          6,999.2 |
|           8 |                  0.93 |           8,768.8 |          10,594.1 |                1.01 |          8,574.0 |          8,728.3 |
|          16 |                  1.42 |          11,406.6 |          12,570.9 |                1.55 |         11,207.8 |         11,422.0 |
|          32 |                  2.13 |          14,926.4 |          17,876.0 |                2.34 |         14,624.0 |         14,974.6 |
|          64 |                  3.27 |          18,807.1 |          24,316.9 |                3.66 |         18,644.1 |         19,212.2 |
|         128 |                  4.15 |          32,117.1 |          40,870.4 |                4.85 |         26,194.8 |         37,102.4 |
|         256 |                  4.99 |          54,099.9 |          68,269.4 |                4.83 |         27,240.0 |         38,025.4 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|           2 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|           4 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|           8 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|          16 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|          32 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|          64 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |
|         128 |          99.9% |            0.1% |           0.0% |        100.0% |           0.0% |          0.0% |
|         256 |         100.0% |            0.0% |           0.0% |        100.0% |           0.0% |          0.0% |

**Output-dominated scenario.** IORails advantage narrows to ~10–17% at c=1–64 and disappears at c=256 (LLMRails 4.99 vs IORails 4.83 req/s). However IORails P99 at c=256 is meaningfully better: 38,025ms vs 68,269ms. No overload fast-fails observed at any concurrency level.

---

### Agent (input 8,000 tokens / output 4,000 tokens / std 20)

| Concurrency | LLMRails tput (req/s) | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails tput (req/s) | IORails P50 (ms) | IORails P99 (ms) |
|-------------|----------------------|-------------------|-------------------|---------------------|------------------|------------------|
|           1 |                  0.45 |           1,380.3 |          10,834.4 |                0.18 |          5,683.1 |         10,647.6 |
|           2 |                  0.83 |           1,559.2 |          10,468.7 |                0.33 |          6,339.4 |         11,911.3 |
|           4 |                  1.43 |           1,901.1 |          14,136.5 |                0.56 |          7,124.7 |         14,488.9 |
|           8 |                  2.19 |           2,594.2 |          16,428.4 |                0.89 |          8,806.7 |         18,622.7 |
|          16 |                  2.73 |           4,625.4 |          21,393.5 |                1.34 |         11,995.7 |         25,077.3 |
|          32 |                  3.06 |           9,021.6 |          27,172.6 |                1.88 |         15,718.6 |         29,941.0 |
|          64 |                  3.26 |          18,119.5 |          33,782.0 |                2.31 |         20,451.0 |         31,668.3 |
|         128 |                  3.19 |          36,900.8 |          57,792.9 |                1.76 |         21,893.2 |         37,735.1 |
|         256 |                  2.92 |          75,748.5 |         116,798.8 |                1.51 |         30,743.8 |         47,461.3 |

| Concurrency | LLMRails Safe% | LLMRails Block% | LLMRails Ovld% | IORails Safe% | IORails Block% | IORails Ovld% |
|-------------|----------------|-----------------|----------------|---------------|----------------|---------------|
|           1 |          93.3% |            6.7% |           0.0% |         92.3% |           7.7% |          0.0% |
|           2 |          94.0% |            6.0% |           0.0% |         88.7% |          11.3% |          0.0% |
|           4 |          91.8% |            8.2% |           0.0% |         92.9% |           7.1% |          0.0% |
|           8 |          92.9% |            7.1% |           0.0% |         92.5% |           7.5% |          0.0% |
|          16 |          90.4% |            9.6% |           0.0% |         91.8% |           8.2% |          0.0% |
|          32 |          91.9% |            8.1% |           0.0% |         93.4% |           6.6% |          0.0% |
|          64 |          91.2% |            8.8% |           0.0% |         95.4% |           4.6% |          0.0% |
|         128 |          93.0% |            7.0% |           0.0% |         92.1% |           7.9% |          0.0% |
|         256 |          91.9% |            8.1% |           0.0% |         91.3% |           8.7% |          0.0% |

**Notable reversal: LLMRails outperforms IORails on throughput at every concurrency level.** LLMRails peaks at 3.26 req/s (c=64) vs IORails 2.31 req/s — LLMRails ~1.4× higher. LLMRails P50 is also consistently lower (1,380ms vs 5,683ms at c=1). IORails recovers on P99 at c=128+ (37,735ms vs 57,793ms at c=128; 47,461ms vs 116,799ms at c=256) due to queue-depth limiting. Hypothesis: IORails' bounded work queue creates head-of-line blocking for very long (8k in / 4k out) requests; LLM inference time so dominates that IORails' overhead disadvantage outweighs its queuing benefit at moderate concurrency.

---

## Scenario Sweeps v3 — P90 Latency (ms)

P90 added per blog plan requirements. Full P50/P99 tables are in the per-scenario sections above.

| Scenario | C | LLMRails P50 | LLMRails P90 | LLMRails P99 | IORails P50 | IORails P90 | IORails P99 |
|----------|---|-------------|-------------|-------------|------------|------------|------------|
| Dialog   |   1 |       461.6 |       477.3 |       488.4 |      406.9 |      421.4 |      431.8 |
| Dialog   |  32 |     2,402.8 |     3,384.2 |     4,157.4 |    1,084.2 |    1,152.8 |    1,225.1 |
| Dialog   | 128 |     8,176.4 |    14,327.4 |    22,214.6 |    2,669.5 |    2,930.0 |    3,128.9 |
| Dialog   | 256 |    17,648.1 |    28,696.2 |    44,964.6 |    5,429.9 |    5,674.7 |    5,916.1 |
| RAG      |   1 |       886.6 |       928.6 |       990.3 |      790.8 |      819.5 |      838.4 |
| RAG      |  32 |     5,420.5 |     8,636.4 |    12,178.5 |    2,883.6 |    3,194.1 |    3,568.7 |
| RAG      | 128 |    22,021.5 |    38,359.0 |    54,540.5 |    8,344.5 |    9,167.3 |    9,785.0 |
| RAG      | 256 |    47,151.6 |    52,652.6 |    57,408.2 |   12,065.6 |   21,481.5 |   22,937.3 |
| Code Gen |   1 |     5,495.2 |     5,521.7 |     5,547.3 |    5,384.9 |    5,413.4 |    5,437.0 |
| Code Gen |  32 |    14,926.4 |    16,017.7 |    17,876.0 |   14,624.0 |   14,793.6 |   14,974.6 |
| Code Gen | 128 |    32,117.1 |    36,952.7 |    40,870.4 |   26,194.8 |   31,167.6 |   37,102.4 |
| Code Gen | 256 |    54,099.9 |    62,691.0 |    68,269.4 |   27,240.0 |   37,526.5 |   38,025.4 |
| Agent    |   1 |     1,380.3 |     3,917.9 |    10,834.4 |    5,683.1 |    8,365.4 |   10,647.6 |
| Agent    |  32 |     9,021.6 |    15,567.6 |    27,172.6 |   15,718.6 |   23,663.0 |   29,941.0 |
| Agent    | 128 |    36,900.8 |    48,490.4 |    57,792.9 |   21,893.2 |   30,726.6 |   37,735.1 |
| Agent    | 256 |    75,748.5 |    89,408.4 |   116,798.8 |   30,743.8 |   41,861.5 |   47,461.3 |

---

## Scenario Sweeps v3 — Analysis

### IORails advantage by scenario

| Scenario | Token profile | IORails peak tput advantage | IORails P99 advantage (c=256) |
|----------|--------------|----------------------------|-------------------------------|
| Dialog   | 100 in / 100 out | **3.3×** (47.8 vs 14.5 req/s) | **7.6×** (5,916ms vs 44,965ms) |
| RAG      | 4k in / 200 out  | **2.7×** (15.2 vs 5.5 req/s)  | **2.5×** (22,937ms vs 57,408ms) |
| Code Gen | 200 in / 2k out  | **~1.1×** (4.85 vs 4.15 req/s) | **1.8×** (38,025ms vs 68,269ms) |
| Agent    | 8k in / 4k out   | **LLMRails wins** (3.26 vs 2.31 req/s) | IORails wins P99 only (47,461ms vs 116,799ms) |

### Key findings vs v2

- **max_tokens fix eliminates unconstrained output**: Dialog OSL mean 100.7 tokens (v2: ~2,982). Safe% is now 100% for dialog/code gen vs the artificially inflated rates in v2.
- **IORails advantage is real but scenario-dependent**: Largest for short balanced workloads (Dialog: 3.3×), shrinks as output length grows (Code Gen: ~1.1×), reverses for very long prompts (Agent: LLMRails wins on throughput).
- **No overload fast-fails observed** in dialog or code gen — unlike v2, where IORails fast-failed 60–70% of dialog requests at c=256. The unconstrained output in v2 made requests take far longer, saturating the IORails queue much earlier.
- **RAG block rate discrepancy** (~15% LLMRails vs ~5% IORails): reflects different aiperf-generated input corpora between runs, not an engine-level difference.

---

## Scenario Sweeps v4 (2026-08-06, 300s per level; lowc 1200s) ← latest
**Hardware:** 8× H100 80GB (Brev node brev-sax6j9j37)
**NIMs:** 6× Nemotron 3 Nano 2.0.10 (GPUs 0–5, ports 8010–8015, nginx round-robin → port 8000) + 2× Nemotron 3.5 Content Safety 2.0.5-variant (GPUs 6–7, ports 8001–8002, nginx → port 8003)
**Guardrails:** branch dev/schilton/iorails-vs-llmrails-benchmarking (tip e642510d2)
**Datasets:** Real-world open datasets (LMSYS-Chat-1M CC BY-NC, MS MARCO Apache-2.0, DS-1000 Apache-2.0, StableToolBench Apache-2.0) + HarmBench unsafe slice (~3%); generated by `benchmark/scripts/blend_datasets.py`
**Key changes vs v3:**
- Data-parallel NIM layout (6× Nano + 2× CS) so downstream models don't bottleneck before IORails/LLMRails overhead is visible
- Real-world datasets replace Shakespeare/synthetic corpora
- Extended benchmark_duration (1200s) for code_gen and agent at c=1, c=2 to ensure ≥200 measurements; c≥4 remains 300s
- Nano bumped from 2.0.8 → 2.0.10

**⚠ Two pre-blog blockers identified — see investigation items below.**

---

### Dialog (input ~100 tokens / output 100 tokens)

Zero errors across all concurrency levels for both engines. Blog-ready.

| Concurrency | LLMRails req/s | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails req/s | IORails P50 (ms) | IORails P99 (ms) | IOR/LLM tput |
|-------------|---------------|-------------------|-------------------|--------------|------------------|------------------|--------------|
|           1 |          1.93 |             518.2 |             550.9 |         2.16 |            461.8 |            490.3 |         1.1× |
|           4 |          6.26 |             635.1 |             859.4 |         8.48 |            470.3 |            507.4 |         1.4× |
|          16 |         13.59 |           1,123.6 |           1,976.0 |        25.33 |            629.2 |            753.9 |         1.9× |
|          64 |         14.39 |           3,980.3 |          10,694.0 |        52.33 |          1,244.2 |          1,421.7 |         3.6× |
|         128 |         14.07 |           8,112.7 |          22,678.2 |        84.10 |          1,521.7 |          1,859.2 |         6.0× |
|         256 |         13.58 |          17,004.0 |          45,842.7 |        89.93 |          2,871.8 |          3,220.9 |     **6.6×** |

**IORails peak:** 89.9 req/s vs LLMRails 14.4 req/s → **6.6× throughput**. P99 at c=256: IORails 3.2s vs LLMRails 45.8s → **−93%**.

---

### RAG (input ~4,000 tokens / output 200 tokens)

Zero errors across all concurrency levels for both engines. Blog-ready.

| Concurrency | LLMRails req/s | LLMRails P50 (ms) | LLMRails P99 (ms) | IORails req/s | IORails P50 (ms) | IORails P99 (ms) | IOR/LLM tput |
|-------------|---------------|-------------------|-------------------|--------------|------------------|------------------|--------------|
|           1 |          1.04 |             989.6 |           1,070.8 |         1.13 |            887.8 |            943.7 |         1.1× |
|           4 |          3.56 |           1,100.2 |           2,621.6 |         4.64 |            862.7 |            933.1 |         1.3× |
|          16 |          5.38 |           2,865.1 |           5,171.7 |        13.63 |          1,180.3 |          1,373.9 |         2.5× |
|          64 |          5.29 |          10,915.7 |          28,283.0 |        26.88 |          2,330.4 |          3,355.8 |         5.1× |
|         128 |          5.11 |          21,189.5 |          60,871.3 |        37.77 |          3,356.9 |          4,152.1 |     **7.4×** |
|         256 |          5.14 |          47,165.3 |          64,733.5 |        37.63 |          6,782.3 |          7,470.1 |         7.3× |

**IORails peak:** 37.8 req/s vs LLMRails 5.4 req/s → **7.4× throughput**. P99 at c=64: IORails 3.4s vs LLMRails 28.3s → **−88%**. IORails saturates at c=128 (same as c=256).

---

### Code Generation (input ~200 tokens / output ~1,800–1,900 tokens)

IORails wins at c=4 through c=128. c=256 suspect due to error spike.

| Concurrency | LLMRails req/s | LLMRails P50 (ms) | LLMRails P99 (ms) | LLM errors | IORails req/s | IORails P50 (ms) | IORails P99 (ms) | IOR errors | IOR/LLM tput |
|-------------|---------------|-------------------|-------------------|-----------|--------------|------------------|------------------|-----------|--------------|
|           1 |          0.16 |           6,467.4 |           6,533.9 |         0 |         0.17 |          6,381.0 |          6,407.8 |         0 |         1.1× |
|           2 |          0.32 |           6,461.7 |           6,633.1 |         0 |         0.35 |          6,382.6 |          6,405.7 |         0 |         1.1× |
|           4 |          0.63 |           6,472.7 |           6,632.0 |         0 |         0.69 |          6,393.3 |          6,416.6 |         0 |         1.1× |
|          16 |          1.94 |           8,278.2 |           9,966.7 |         0 |         2.08 |          8,099.2 |          9,220.4 |         0 |         1.1× |
|          64 |          3.96 |          15,842.1 |          23,331.4 |         0 |         4.40 |         15,225.8 |         16,397.4 |         0 |         1.1× |
|         128 |          4.58 |          29,871.8 |          38,470.6 |        14 |         5.96 |         21,386.5 |         31,334.3 |         7 |    **+30%** |
|     256 ⚠  |          5.05 |          46,214.2 |          71,485.9 |         5 |         5.94 |         23,596.7 |         34,765.3 |     1,906 |      suspect |

**c=256 IORails not reliable** — 1,906 errors (vs 5 for LLMRails) indicate backend saturation. Use c=128 as the headline: **+30% throughput, −31% P99**.

---

### Code Generation — Low Concurrency (1200s runs)

| Concurrency | LLMRails req/s | LLMRails count | LLMRails P50 (ms) | IORails req/s | IORails count | IORails P50 (ms) |
|-------------|---------------|---------------|-------------------|--------------|--------------|------------------|
|           1 |          0.160 |           191 |           6,467.4 |        0.173 |          207 |          6,331.4 |
|           2 |          0.322 |           386 |           6,461.7 |        0.350 |          420 |          6,332.9 |

Both engines behave nearly identically at low concurrency for code_gen — LLM inference time dominates completely.

---

### Agent (input ~8,000 tokens / output target 4,000 tokens) — ⚠ UNDER INVESTIGATION

**Do not use for blog.** LLMRails generates ~560 tokens/request; IORails generates ~1,750 tokens/request against a 4,000-token target. The low-concurrency "LLMRails wins on req/s" finding is an artifact of this truncation, not a genuine engine performance difference. Additionally, IORails hits significant error rates at c=128+ (213 errors at c=128; 1,097 at c=256).

| Concurrency | LLMRails req/s | LLMRails tok/req | LLM errors | IORails req/s | IORails tok/req | IOR errors |
|-------------|---------------|-----------------|-----------|--------------|----------------|-----------|
|           1 |          0.422 |             556 |         0 |        0.155 |           1776 |         0 |
|           2 |          0.783 |             597 |         0 |        0.331 |           1698 |         0 |
|           4 |          1.532 |             538 |         0 |        0.599 |           1793 |         0 |
|          32 |          3.099 |             556 |         0 |        2.752 |           1775 |         0 |
|          64 |          3.226 |             583 |        12 |        3.775 |           1721 |        35 |
|         128 |          3.081 |             534 |        27 |        4.720 |           1543 |       213 |
|         256 |          2.944 |             579 |         1 |        4.361 |           1490 |     1,097 |

**tok/req discrepancy:** LLMRails generates ~3.1× fewer tokens than IORails (target: 4,000). Root cause unknown — hypotheses: (a) LLMRails' serial content-safety-on-8K-input path exhausts the time budget before generation completes; (b) residual NGUARD-873 max_tokens propagation issue specific to LLMRails + long ISL. Flagged to Tim Gasser 2026-08-07.

**IORails error spike:** 213 errors at c=128 (13%) and 1,097 at c=256 (46%) — backend saturation from IORails dispatching far more requests than LLMRails at equal concurrency.

---

### v4 Pre-Blog Investigation Items

**1. LLMRails agent output truncation (blocker)**
LLMRails agent scenario generates ~560 tokens/request against a 4,000-token target. IORails generates ~1,750. This makes the low-c "LLMRails wins" comparison unfair — LLMRails is faster because it returns shorter responses. Diagnosis and fix required before agent results can be published.

**2. IORails c=256 saturation for long-output scenarios (blocker for c=256 data)**
IORails agent c=256: 1,097 errors; code_gen c=256: 1,906 errors. LLMRails: essentially zero at both. IORails is dispatching far more requests at equal concurrency (because it's faster), overwhelming the NIM pool. The reliable IORails ceiling for long-output scenarios is c=128. c=256 data for agent and code_gen should be excluded from the blog.

**v4 blog-ready results:**
- **Dialog:** Clean win. Use c=256 headline: 6.6× throughput, −93% P99. Zero errors both engines.
- **RAG:** Clean win. Use c=128 headline: 7.4× throughput, −88% P99 at c=64. Zero errors both engines.
- **Code Gen:** IORails wins. Use c=128 headline: +30% throughput, −31% P99. Drop c=256 data.
- **Agent:** Hold. Diagnosis pending.

---

## Scenario Sweeps v5 (2026-08-15, 300s per level; lowc 1200s) ← latest
**Hardware:** 8× H100 80GB (Brev node brev-sax6j9j37)
**NIMs:** 6× Nemotron 3 Nano 2.0.10 (GPUs 0–5, ports 8010–8015, nginx → port 8000) + 2× Nemotron 3.5 Content Safety 2.0.5-variant (GPUs 6–7, ports 8001–8002, nginx → port 8003)
**Guardrails:** branch dev/schilton/iorails-vs-llmrails-benchmarking (tip 988757da3)
**Key changes vs v4:**
- Real-world datasets wired into aiperf via `--input-file` / `--custom-dataset-type single_turn` (v4 inadvertently used synthetic token distributions)
- Datasets regenerated: dialog (LMSYS-Chat-1M CC BY-NC, ISL 50–300 tok), rag (MS MARCO v1.1 Apache-2.0, ISL 500–2,000 tok), code_gen (DS-1000 Apache-2.0), agent (NousResearch/hermes-function-calling-v1 Apache-2.0; StableToolBench migrated to leaderboard-only)
- `use_server_token_count: true` — client-side tokenizer bypassed; server reports token counts
- RAG ISL target corrected from 4,000 tok (synthetic) to ~1,000 tok (actual MS MARCO distribution)

**Streaming sweep attempted but failed** — Guardrails server rejects all streaming requests unconditionally; TTFT/ITL metrics not available. Filed as a separate blocker.

---

### Dialog (LMSYS-Chat-1M real data, ISL ~100 tok / OSL 100 tok)

Zero IORails errors across all concurrency levels. Blog-ready.

| Concurrency | LLMRails req/s | LLMRails P50 (ms) | LLMRails P99 (ms) | LLM err | IORails req/s | IORails P50 (ms) | IORails P99 (ms) | IO err | IOR/LLM tput |
|-------------|---------------|-------------------|-------------------|---------|--------------|------------------|------------------|--------|--------------|
|           1 |          2.294 |             516 |             543 |       1 |         2.582 |            462 |            474 |      0 |         1.1× |
|           2 |          4.380 |             526 |             625 |       3 |         5.150 |            460 |            486 |      0 |         1.2× |
|           4 |          7.497 |             586 |             986 |       4 |         9.944 |            472 |            505 |      0 |         1.3× |
|           8 |         11.807 |             729 |           1,234 |       8 |        18.204 |            504 |            592 |      0 |         1.5× |
|          16 |         14.753 |           1,139 |           1,967 |       8 |        29.824 |            614 |            752 |      0 |         2.0× |
|          32 |         14.387 |           2,478 |           3,700 |       9 |        42.409 |            859 |          1,071 |      0 |         2.9× |
|          64 |         15.808 |           3,683 |          10,729 |       9 |        56.378 |          1,298 |          1,645 |      0 |         3.6× |
|         128 |         15.632 |           7,339 |          22,093 |      10 |        88.697 |          1,631 |          2,133 |      0 |         5.7× |
|         256 |         14.989 |          14,782 |          48,715 |      10 |       105.248 |          2,866 |          3,280 |      0 |     **7.0×** |

**IORails peak:** 105.2 req/s @ c=256 vs LLMRails 15.8 req/s @ c=64 → **6.66× throughput**. P99 at c=256: IORails 3.3s vs LLMRails 48.7s → **−93%** (14.85×). IORails scales continuously through c=256 with zero errors.

**v4 comparison:** v4 dialog was 6.6×; v5 is 6.66× — near-identical. Real-data dialog workload confirms synthetic result. High confidence.

---

### RAG (MS MARCO v1.1 real data, ISL ~1,000 tok / OSL 200 tok)

Note: v4 used synthetic 4,000-token RAG prompts; v5 uses real MS MARCO queries + passages (actual distribution: ISL 500–2,000 tok, median ~1,000 tok). This explains the throughput ratio change vs v4.

| Concurrency | LLMRails req/s | LLMRails P50 (ms) | LLMRails P99 (ms) | LLM err | IORails req/s | IORails P50 (ms) | IORails P99 (ms) | IO err | IOR/LLM tput |
|-------------|---------------|-------------------|-------------------|---------|--------------|------------------|------------------|--------|--------------|
|           1 |          1.166 |             899 |             912 |       1 |         1.294 |            824 |            832 |      0 |         1.1× |
|           2 |          2.278 |             905 |           1,021 |       1 |         2.608 |            824 |            850 |      0 |         1.1× |
|           4 |          4.192 |             968 |           1,240 |       1 |         5.130 |            828 |            873 |      0 |         1.2× |
|           8 |          7.323 |           1,101 |           1,526 |       1 |         9.198 |            877 |          1,185 |      0 |         1.3× |
|          16 |          9.664 |           1,614 |           3,836 |       1 |        14.868 |          1,133 |          1,328 |      0 |         1.5× |
|          32 |         10.660 |           2,910 |           5,689 |       1 |        21.970 |          1,515 |          1,942 |      0 |         2.1× |
|          64 |         10.739 |           6,120 |          10,965 |       1 |        29.201 |          2,256 |          3,092 |      0 |         2.7× |
|         128 |         11.235 |          10,326 |          26,765 |       2 |        40.301 |          3,320 |          4,030 |      0 |     **3.6×** |
|     256 ⚠  |         10.714 |          24,612 |          35,172 |      21 |        39.937 |          6,571 |          7,756 |      0 |         3.7× |

**IORails peak:** 40.3 req/s @ c=128 vs LLMRails 11.2 req/s @ c=128 → **3.59× throughput**. P99 at c=128: IORails 4.0s vs LLMRails 26.8s → **6.64× lower**. Both engines plateau at c=128; neither benefits from c=256. LLMRails accumulates 21 errors at c=256 (IORails: 0).

**v4 comparison: 7.4× → 3.59% — significant drop.** Real MS MARCO queries are shorter and more uniform than the synthetic 4,000-token prompts used in v4 (actual median ~1,000 tok vs synthetic 4,000 tok). Shorter, more uniform input reduces the queuing pressure that IORails resolves, narrowing the gap. This is the correct real-world figure.

---

### Code Generation (DS-1000 real data, ISL ~200–800 tok / OSL 2,000 tok)

c=1 and c=2 from `_lowc` directories (1200s runs). c=256 IORails excluded — 1,741 errors.

| Concurrency | LLMRails req/s | LLMRails P50 (ms) | LLMRails P99 (ms) | LLM err | IORails req/s | IORails P50 (ms) | IORails P99 (ms) | IO err | IOR/LLM tput |
|-------------|---------------|-------------------|-------------------|---------|--------------|------------------|------------------|--------|--------------|
|          1* |          0.228 |           4,426 |           6,552 |       0 |         0.220 |          4,691 |          6,427 |      0 |         1.0× |
|          2* |          0.439 |           4,683 |           6,562 |       0 |         0.436 |          4,706 |          6,428 |      0 |         1.0× |
|           4 |          0.889 |           4,485 |           6,745 |       0 |         0.841 |          5,028 |          6,944 |      0 |         0.9× |
|           8 |          1.605 |           4,896 |           7,628 |       0 |         1.547 |          5,489 |          7,712 |      0 |         1.0× |
|          16 |          2.699 |           5,895 |           9,761 |       0 |         2.667 |          6,240 |          9,345 |      0 |         1.0× |
|          32 |          4.043 |           7,963 |          13,031 |       0 |         4.034 |          8,078 |         12,214 |      0 |         1.0× |
|          64 |          5.617 |          11,327 |          19,554 |       1 |         5.762 |         11,215 |         16,463 |      0 |         1.0× |
|         128 |          7.230 |          17,229 |          33,716 |       4 |         7.697 |         16,435 |         24,061 |      0 |     **+6%** |
|     256 ⚠  |          7.618 |          30,138 |          64,725 |      12 |         7.940 |         20,499 |         34,470 |  1,741 |      suspect |

**Clean operating point: c=128.** IORails 7.7 req/s vs LLMRails 7.2 req/s → **+6% throughput, −29% P99**. Engines are at parity for output-dominated workloads; LLM inference time dominates. Do not use IORails c=256 (1,741 errors — backend saturation).

---

### Code Generation — Low Concurrency (1200s runs)

| Concurrency | LLMRails req/s | LLMRails count | LLMRails P50 (ms) | IORails req/s | IORails count | IORails P50 (ms) |
|-------------|---------------|---------------|-------------------|--------------|--------------|------------------|
|           1 |          0.228 |           274 |           4,426 |         0.220 |          264 |          4,691 |
|           2 |          0.439 |           528 |           4,683 |         0.436 |          524 |          4,706 |

Near-identical at low concurrency — LLM dominates completely.

---

### Agent (Hermes function-calling real data, ISL ~1,000–12,000 tok / OSL 4,000 tok)

c=1 and c=2 from `_lowc` directories (1200s runs). IORails error onset at c=64; significant errors at c=128+.

**Note:** Agent dataset changed from StableToolBench (v4, ~8,000 tok ISL) to NousResearch/hermes-function-calling-v1 (v5, ISL 1,000–12,000 tok). Output size discrepancy present: IORails generates ~35% more tokens at low concurrency vs LLMRails, suggesting different generation behavior between engines for this dataset. This is the same pattern investigated in v4 but now with a different dataset.

| Concurrency | LLMRails req/s | LLMRails P50 (ms) | LLMRails P99 (ms) | LLM err | IORails req/s | IORails P50 (ms) | IORails P99 (ms) | IO err | IOR/LLM tput |
|-------------|---------------|-------------------|-------------------|---------|--------------|------------------|------------------|--------|--------------|
|          1* |          0.393 |           2,212 |           9,496 |       0 |         0.341 |          2,443 |         12,449 |      0 |         0.9× |
|          2* |          0.801 |           2,131 |           9,062 |       0 |         0.635 |          2,511 |         12,672 |      0 |         0.8× |
|           4 |          1.496 |           2,282 |          10,097 |       0 |         1.308 |          2,508 |         13,205 |      0 |         0.9× |
|           8 |          2.650 |           2,524 |          12,300 |       0 |         2.267 |          2,744 |         15,363 |      0 |         0.9× |
|          16 |          4.453 |           3,033 |          14,404 |       1 |         3.719 |          3,305 |         18,945 |      0 |         0.8× |
|          32 |          6.118 |           4,430 |          20,494 |       1 |         5.844 |          4,116 |         24,252 |      0 |         1.0× |
|      64 ⚠  |          6.731 |           8,714 |          23,906 |       1 |         8.374 |          5,700 |         27,956 |     61 |         1.2× |
|     128 ⚠  |          6.876 |          17,707 |          35,665 |       9 |        11.128 |          9,104 |         27,467 |    159 |         1.6× |
|     256 ⚠  |          6.830 |          34,853 |          57,924 |      14 |        10.783 |         15,382 |         35,227 |  1,108 |         1.6× |

**LLMRails wins at c=1–c=32 (clean operating points, zero errors both engines).** IORails takes the throughput lead at c=64+ but with significant error rates (61–1,108). Last clean parity point: c=32 (6.12 vs 5.84 req/s, ~1.05× LLMRails). Agent scenario requires Tim's diagnosis of IORails error onset and output-size discrepancy before blog use.

---

### v5 Pre-Blog Investigation Items

**1. RAG throughput ratio drop (7.4× → 3.59×) — informational, not a blocker**
Real MS MARCO ISL (~1,000 tok median) is substantially shorter than the v4 synthetic prompts (4,000 tok). Shorter, more uniform inputs reduce the IORails queuing advantage. The 3.59× figure is the correct real-world RAG result. Recommend updating blog copy accordingly.

**2. Agent IORails error onset at c=64 (blocker)**
Same pattern as v4 — IORails dispatches more requests than LLMRails, eventually saturating the NIM pool. Agent output-size discrepancy (IORails generates ~35% more tokens at low concurrency) persists from v4 with a different dataset, suggesting it is an engine-level behavior, not a dataset artifact. Pending Tim's diagnosis.

**3. Code Gen IORails c=256 saturation (informational)**
1,741 IORails errors at c=256; c=128 is the reliable operating point. Same as v4.

**4. Streaming metrics unavailable**
Guardrails server rejects all streaming requests unconditionally (api.py:625 — condition fires for any streaming request regardless of tools). TTFT/ITL cannot be measured without a server-side fix.

**v5 blog-ready results:**
- **Dialog:** 6.66× throughput, −93% P99 at c=256. Confirmed with real data. Zero IORails errors.
- **RAG:** 3.59× throughput, −85% P99 at c=128. Real-world figure (lower than v4 synthetic due to shorter real ISL).
- **Code Gen:** ~1.06× throughput at c=128, −29% P99. Parity scenario. Exclude IORails c=256.
- **Agent:** Hold. IORails error onset + output-size discrepancy pending diagnosis.

---

## Methodology notes
- OSL classification thresholds (1–5 = overload, 6–20 = blocked, >20 = safe) were validated against the OSL distribution: no records fall in the 12–20 range, confirming a clean natural separator
- RAG/Agent cross-engine block rate comparisons are not reliable: aiperf regenerates the input corpus independently for each engine run, so different Shakespeare passages are sampled
- `benchmark_duration: 300` per concurrency level (updated from 120s to amortize transient artifacts)
