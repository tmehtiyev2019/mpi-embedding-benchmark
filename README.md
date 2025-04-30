mpi-embedding-benchmark/
├── mpi_embedding_benchmark.py
├── combined_charts.py
├── run_mpi.bat
├── requirements.txt
├── benchmark_results/
├── mpi_results/
├── output/
├── combined_comparison/
├── final_charts/
├── .gitignore
└── README.md


# MPI Embedding Benchmark

This project benchmarks document embedding inference using SentenceTransformers with MPI-based parallelism. The goal is to explore speedup, throughput, and efficiency when processing large batches of documents across multiple CPU processes.

## Features
- Parallel embedding using `mpi4py`
- Benchmarking with synthetic datasets
- Charts for speedup, throughput, efficiency
- Automation via `.bat` script

## Setup

### Requirements
- Python 3.8+
- `mpi4py`
- `sentence-transformers`
- `matplotlib`
- `numpy`
- OpenMPI or Microsoft MPI installed

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Benchmarks
Run the full benchmark pipeline for 2, 4, and 8 MPI processes:

```bash
run_mpi.bat
```


This will:
* Generate synthetic document datasets
* Run benchmarks at multiple process counts
* Save results to `mpi_results/`
* Generate combined comparison charts in `combined_comparison/`

## Output

* Per-process benchmarking results in `mpi_results/`
* Summary charts in `final_charts/`
* Cross-comparison plots in `combined_comparison/`