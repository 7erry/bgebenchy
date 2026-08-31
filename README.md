# bgebenchy

Motion-shaped catalog embedding benchmark. Compares [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and [voyage-4-large](https://huggingface.co/voyageai/voyage-4-large) at **10,000 documents**, **1024-d fp16**, **23 filter fields**, **3 full-text fields**, targeting **95% recall@10**.

Interactive layout of these same numbers: Cursor canvas `bge-large-10k-benchmark`.

## Results

Same corpus, cosine HNSW (usearch, M=16, `ef_construction=128`, `ef_search=16`). Open-loop load: 5s warmup + 20s measure. Zero late starts at 50 / 75 / 100 RPS.

| | BGE-large-en-v1.5 | voyage-4-large |
|---|---:|---:|
| Recall@10 | **95.45%** | **95.65%** |
| p99 at 100 RPS | 1.70 ms | 1.55 ms |
| Index build | 0.24 s | 0.22 s |
| HNSW memory | 32.1 MB | 32.1 MB |

Latency is **local in-process ANN on an Apple M4 Pro**. It does not include MongoDB, mongot, network, or Lucene filter / full-text cost.

### p99 latency by model

Matches the canvas bar chart: p99 (ms) at the 95% recall operating point.

```mermaid
xychart-beta
    title p99 latency at 95% recall (ms)
    x-axis [50 RPS, 75 RPS, 100 RPS]
    y-axis "p99 (ms)" 0 --> 2.6
    bar [2.468, 1.789, 1.695]
    line [1.154, 1.598, 1.548]
```

- Bars: BGE-large-en-v1.5 p99
- Line: voyage-4-large p99

<svg viewBox="0 0 720 280" width="100%" role="img" aria-label="Grouped bar chart of p99 latency in milliseconds at 50, 75, and 100 requests per second for BGE-large-en-v1.5 and voyage-4-large">
  <rect x="0" y="0" width="720" height="280" fill="#0d1117"/>
  <text x="360" y="28" fill="#e6edf3" font-size="15" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="middle">p99 latency at 95% recall (ms)</text>
  <text x="28" y="150" fill="#8b949e" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="middle" transform="rotate(-90 28 150)">p99 (ms)</text>
  <line x1="64" y1="48" x2="64" y2="220" stroke="#30363d"/>
  <line x1="64" y1="220" x2="680" y2="220" stroke="#30363d"/>
  <line x1="64" y1="177" x2="680" y2="177" stroke="#21262d"/>
  <line x1="64" y1="134" x2="680" y2="134" stroke="#21262d"/>
  <line x1="64" y1="91" x2="680" y2="91" stroke="#21262d"/>
  <text x="56" y="224" fill="#8b949e" font-size="10" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="end">0</text>
  <text x="56" y="181" fill="#8b949e" font-size="10" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="end">0.65</text>
  <text x="56" y="138" fill="#8b949e" font-size="10" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="end">1.30</text>
  <text x="56" y="95" fill="#8b949e" font-size="10" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="end">1.95</text>
  <text x="56" y="56" fill="#8b949e" font-size="10" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="end">2.60</text>
  <!-- 50 RPS: BGE 2.468, Voyage 1.154 -->
  <rect x="118" y="56.6" width="44" height="163.4" fill="#388bfd"/>
  <rect x="168" y="143.7" width="44" height="76.3" fill="#3fb950"/>
  <text x="165" y="238" fill="#8b949e" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="middle">50 RPS</text>
  <!-- 75 RPS: BGE 1.789, Voyage 1.598 -->
  <rect x="318" y="101.6" width="44" height="118.4" fill="#388bfd"/>
  <rect x="368" y="114.3" width="44" height="105.7" fill="#3fb950"/>
  <text x="365" y="238" fill="#8b949e" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="middle">75 RPS</text>
  <!-- 100 RPS: BGE 1.695, Voyage 1.548 -->
  <rect x="518" y="107.8" width="44" height="112.2" fill="#388bfd"/>
  <rect x="568" y="117.6" width="44" height="102.4" fill="#3fb950"/>
  <text x="565" y="238" fill="#8b949e" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif" text-anchor="middle">100 RPS</text>
  <rect x="230" y="256" width="10" height="10" fill="#388bfd"/>
  <text x="246" y="265" fill="#e6edf3" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif">BGE-large-en-v1.5</text>
  <rect x="400" y="256" width="10" height="10" fill="#3fb950"/>
  <text x="416" y="265" fill="#e6edf3" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif">voyage-4-large</text>
</svg>

### Latency (ms) at 95% recall, `ef_search=16`

| Model | RPS | p50 | p90 | p95 | p99 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| BGE-large-en-v1.5 | 50 | 0.271 | 0.614 | 0.791 | 2.468 | 0.9545 |
| BGE-large-en-v1.5 | 75 | 0.260 | 0.606 | 0.775 | 1.789 | 0.9545 |
| BGE-large-en-v1.5 | 100 | 0.245 | 0.473 | 0.715 | 1.695 | 0.9545 |
| voyage-4-large | 50 | 0.501 | 0.669 | 0.767 | 1.154 | 0.9565 |
| voyage-4-large | 75 | 0.492 | 0.711 | 0.894 | 1.598 | 0.9565 |
| voyage-4-large | 100 | 0.507 | 0.732 | 0.868 | 1.548 | 0.9565 |

JSON copies: [`stored_results/bge-large-en-v1.5.json`](stored_results/bge-large-en-v1.5.json), [`stored_results/voyage-4-large.json`](stored_results/voyage-4-large.json).

## Index configuration

**23 filter fields:** `productNumber`, `vendorName`, `vendorPartNumber`, `UserTypeID`, `ParentID`, `MotionId`, `StepId`, `UNSPSC`, `ManufacturerID`, `mfr_name_NM`, `product_type_LOV`, `Active`, `WebStatus`, `ItemUOM`, `PGC_CODE`, `bearing_type_LOV`, `cage_type_LOV`, `parent_id_ID`, `eCOS_PGC`, `Prop65`, `company_name`, `ItemNumber`, `meta.source`

**3 full-text fields:** `WebProductDescription`, `DerivedProductDescription`, `descriptionKeywords`

Vector path: `embedding`, 1024 dimensions, cosine, pre-quantized fp16 (`quantization: none`). Voyage used `input_type` `document` / `query`; BGE used the official query prefix.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m benchmark.run --model bge-large-en-v1.5 --docs 10000 --queries 200 \
  --rps 50,75,100 --data-dir data/bge-large-en-v1.5 --results-dir results/bge-large-en-v1.5

# Voyage requires VOYAGE_API_KEY in the environment (or a gitignored .env)
python -m benchmark.run --model voyage-4-large --docs 10000 --queries 200 \
  --rps 50,75,100 --data-dir data/voyage-4-large --results-dir results/voyage-4-large
```
