# AnchorANN

AnchorANN is a private approximate nearest neighbor search method. It:

1. splits the vectors into small equal-size clusters,
2. searches an NSG over the cluster representatives locally on the client, and
3. fetches the selected clusters in one batch with Piano PIR.

This repository contains the Go implementation, dataset preparation scripts, the
AnchorANN measurement runner, the CSVs behind the paper's figures, and the
scripts that draw those figures.

## Contents

| Path | Description |
| --- | --- |
| `anchorann/` | AnchorANN (clustering, our NSG implementation in `anchorann/nsg`, cluster PIR, search) |
| `pianopir/`, `pirclient/`, `rpc/` | Piano PIR and its RPC layer, derived from [Pacmann](https://eprint.iacr.org/2024/1600) |
| `scripts/` | Dataset download/conversion and the measurement runner (`benchmark.py`) |
| `paper/data/` | CSVs plotted in the paper |
| `paper/scripts/` | Figure scripts and the LAN/WAN network model |
| `Dockerfile` | Environment for building, testing and plotting |

## Quick start: redraw the figures

The figure scripts read only `paper/data/*.csv`. No dataset is needed.

```bash
docker build -t anchorann .
docker run --rm -v "$PWD:/artifact" anchorann bash -c \
  'for s in paper/scripts/plot_*.py; do python3 "$s"; done'
```

The PDFs are written to `paper/figures/`.

| Figure | Script | Input CSV |
| --- | --- | --- |
| `RecallQPS.pdf` | `plot_recall_qps.py` | `recall_qps.csv`, `recall_qps_amortized.csv` |
| `OnlineBreakdownTimes.pdf` | `plot_online_breakdown_times.py` | `sift1m_wan_online_breakdown.csv` |
| `ClusterSize.pdf` | `plot_cluster_size.py` | `sift1m_cluster_size.csv` |
| `Ablation.pdf` | `plot_ablation.py` | `sift1m_ablation.csv` |
| `RoundsPIR.pdf` | `plot_rounds_pir.py` | `sift1m_wan_rounds_pir.csv` |

## Local setup

Requirements: Linux, Go 1.22.1+, Python 3.12. No cgo or native ANN library is needed.

```bash
go build ./... && go test ./...
python3 -m pip install -r requirements.txt
```

## Reproducing the measurements

The commands below use a local Go toolchain. In Docker, the image has no Go,
but the four commands are on `PATH`, so replace `go run ./anchorann/cmd/<name>`
with `<name>`. For example:

```bash
docker run --rm -v "$PWD:/artifact" anchorann \
  cluster -input data/SIFT1M/base.bvecs -limit 1000000 -s 32 ...
```

`scripts/datasets.py` and `scripts/benchmark.py` run the same way inside the
image (`docker run --rm -v "$PWD:/artifact" anchorann python3 scripts/...`).

### 1. Prepare datasets

Dataset files are not included.

```bash
python3 scripts/datasets.py download SIFT1M --directory data/SIFT1M
python3 scripts/datasets.py download DEEP1M --directory data/DEEP1M
python3 scripts/datasets.py download GIST1M --directory data/GIST1M
```

Each directory then holds `base.{b,f}vecs`, `query.{b,f}vecs` and `gt.ivecs`.

- SIFT1M is the first 1M vectors of BigANN.
- DEEP1M is the first 1M vectors of Deep1B. Its exact L2 ground truth is
  recomputed, which takes a while.
- To convert other formats (`.npy`, `.fbin`, `.ibin`), use
  `python3 scripts/datasets.py convert <in> <out>`.

<details>
<summary>MS MARCO (optional)</summary>

```bash
python3 -m pip install -r scripts/requirements-msmarco.txt   # also needs liblz4
python3 scripts/download_msmarco.py --directory data/MSMARCO   # ~200 GB of disk
python3 scripts/prepare_msmarco.py base         --directory data/MSMARCO
python3 scripts/prepare_msmarco.py queries      --directory data/MSMARCO
python3 scripts/prepare_msmarco.py ground-truth --directory data/MSMARCO
python3 scripts/prepare_msmarco_pca.py --source data/MSMARCO --output data/MSMARCO-PCA
```

- The passage vectors are the TAS-B vectors in `macavaney/msmarco-passage.tasb.flex`
  at `87817c2`. The queries are embedded with
  `sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco` at `a03882c`.
- The 31 largest passage IDs are dropped, leaving 8,841,792 vectors, which is
  divisible by 32.
- The PCA step produces a 193-dimensional input: 192 principal components plus
  one L2 augmentation.
- The paper CSVs do not contain MS MARCO results yet.

</details>

### 2. Build the index (server-side preprocessing)

Example for SIFT1M with cluster size s = 32. N must be divisible by s.

```bash
mkdir -p data/SIFT1M/index
go run ./anchorann/cmd/cluster \
  -input data/SIFT1M/base.bvecs -limit 1000000 -s 32 \
  -method common -iterations 6 -seed 711 -workers 20 \
  -output data/SIFT1M/index/clusters.anchor
go run ./anchorann/cmd/build-bundle \
  -artifact data/SIFT1M/index/clusters.anchor \
  -output data/SIFT1M/index/anchors.bundle
```

- These are the settings used in the paper (capacity-constrained Lloyd,
  6 iterations). Clustering SIFT1M takes about 50 minutes with 20 workers.
- Keep `-workers` at 2 or more. With `-workers 1`, `cluster` switches to an
  exact assignment solver that is far slower at this scale.
- `build-bundle` builds the NSG over the representatives. `-workers` does not
  change its output.

### 3. Run the search

```bash
python3 scripts/benchmark.py --dataset SIFT1M \
  --base data/SIFT1M/base.bvecs --queries data/SIFT1M/query.bvecs \
  --gt data/SIFT1M/gt.ivecs \
  --artifact data/SIFT1M/index/clusters.anchor \
  --bundle data/SIFT1M/index/anchors.bundle \
  --probes 4,8,16,32,64,128 --l-search 128 --size 32 \
  --server-cpu 0 --client-cpu 1 \
  --output results/SIFT1M
```

- `--server-cpu` and `--client-cpu` pin the two processes with `taskset`, as in
  the paper. Choose CPUs on different physical cores (check with `lscpu -e`).
  Without them, the processes are not pinned.

- The runner starts the server, runs each point with fresh client hints, and
  stops the server when it finishes.
- `--output` must be a new directory. `results/SIFT1M/model/` receives CSVs in
  the same format as `paper/data/recall_qps*.csv`.
- As in the paper, the client and server run on one machine, each with
  `GOMAXPROCS=1`. LAN/WAN time is then added by the network model below.
- Online query time excludes the bundle download and hint construction. Hint
  regeneration is counted only in the amortized QPS.

## Comparing with other methods

Baseline implementations are **not included**. To compare against another
method, connect it through the following interface.

| Item | Specification |
| --- | --- |
| Input | fvecs (float32), bvecs (uint8), ivecs (int32, zero-based, nearest first). Use the same base/query order and ground truth as above. |
| Per-query measurements | client time, server time, communication rounds, uplink bytes, downlink bytes |
| Hint regeneration | count, time, downlink bytes, interval in queries (0 if not needed) |
| Accuracy | 1-Recall@10: fraction of queries whose true nearest neighbor is in the returned top 10 |
| Output | JSON for `paper/scripts/network_model.py` (see the example written by `scripts/benchmark.py`), which produces CSVs in the `paper/data/` format. The plot scripts can then draw them directly. |

Network model, with RTT r and bandwidth B (bit/s):

```
t_network     = rounds * r + (up_bytes + down_bytes) / (B / 8)
QPS           = 1 / mean(t_processing + t_network)
t_regen       = (regen_seconds + regen_down_bytes / (B / 8)) / regen_count
QPS_amortized = 1 / (mean_online + t_regen / interval_queries)
```

LAN: r = 5 ms, B = 1 Gbps. WAN: r = 50 ms, B = 100 Mbps.

### Baselines used in the paper

| Method | Source | Main parameters |
| --- | --- | --- |
| Pacmann | [privsearch/private-search-temp](https://github.com/privsearch/private-search-temp) @ `99654ad` | degree 32; step = 8–28; parallel = 3; 16 batch partitions |
| Amora | [meander-wqh/Amora](https://github.com/meander-wqh/Amora) @ `221f66b` | T = 1–6; threads = 1 |
| NSG (non-private) | [ZJULearning/nsg](https://github.com/ZJULearning/nsg) @ `5ec8fad` | L/R/C: SIFT1M 40/50/500, GIST1M 60/70/500; search L = 10–1280 |
| kNN graph for NSG | [ZJULearning/efanna_graph](https://github.com/ZJULearning/efanna_graph) @ `6d741ad` | K/L/iter/S/R: SIFT1M 200/200/10/10/100, GIST1M 400/400/12/15/100 |

Check the upstream licenses before redistributing baseline code.

## License

MIT. The PIR components (`pianopir/`, `pirclient/`, `rpc/`) are derived from
Pacmann, and their original notice `Copyright 2024 "Privsearch"` is kept in
`LICENSE`. Third-party Go modules and the datasets keep their own licenses.
