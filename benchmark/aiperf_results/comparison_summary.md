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

## Scenario Sweeps v2 (2026-07-22–23, 300s per level) ← canonical
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

## Methodology notes
- OSL classification thresholds (1–5 = overload, 6–20 = blocked, >20 = safe) were validated against the OSL distribution: no records fall in the 12–20 range, confirming a clean natural separator
- RAG/Agent cross-engine block rate comparisons are not reliable: aiperf regenerates the input corpus independently for each engine run, so different Shakespeare passages are sampled
- `benchmark_duration: 300` per concurrency level (updated from 120s to amortize transient artifacts)
