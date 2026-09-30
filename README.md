# CUDA-Accelerated Neural Network for Power Grid Load Forecasting

**Author:** Stathis Koulofotias

**Date:** August - September 2026

**License:** MIT

---

## Overview

A custom machine learning architecture written in **C and CUDA C**, designed from scratch to demonstrate massive parallelization in neural network training. This project features:

- **Massive parallelization :** Hardware-accelerated training using a **parallel random search** approach on the GPU.
- **Custom build :** Custom multilayer perceptron implementation with forward pass, backpropagation, and gradient descent, built without purely with C and CUDA C.
- **Automated ADMIE data pipeline :** Integrated python module for asynchronous fetching and in-memory parsing of real-time SCADA system load reports.
- **High-speed CPU inference :** Isolated C runtime for low-latency forward pass evaluations, using the winning model's weights.
- **Full Automation :** Modular compilation, dataset generation, and execution via `Makefile`.

This is an educational proof-of-concept bridging low-level parallel computing with power systems engineering. It aims to demonstrate massive GPU acceleration in neural networks while exploring the fundamental operational principles, load patterns, and structural dynamics that govern an electrical grid.

---

## Key Features & Architecture

The system is based on a **modular, multi-stage architecture**, combining specialized environments:

### Data Ingestion Pipeline 
`make_logs.py`

To bridge raw external grid data with the low-level C dataset structures, the system uses an organized fetching structure:
- **Asynchronous Execution:** Utilizes `ThreadPoolExecutor` (8 threads) to download multiple Excel files simultaneously.
- **In-Memory Processing:** Uses `io.BytesIO` to read and parse Excel sheets directly in RAM, bypassing slow disk I/O.
- **SCADA Integration:** Targets the Independent Power Transmission Operator (ADMIE) API (`RealTimeSCADASystemLoad`) to extract 24-hour load figures and format them into a flattened `.txt` dataset.

### CUDA Parallel Trainer
`train_neural.cu`

- Spawns 16,384 independent threads on the GPU.
- Each thread initializes a distinct neural network with a unique random seed.
- Hardware-accelerated batch processing: all networks execute forward passes, backpropagation, and weight updates concurrently on the same dataset.
- Evaluates Mean Squared Error (MSE) and dynamically exports the weights of the single best-performing network to `weights.txt`.

### Fast CPU Inference Engine
`run_neural.c`

- Reads the optimized parameters from the GPU output.
- Executes isolated predictions based on user input for current grid conditions.
- Strictly uncoupled from the training logic for maximum execution speed and minimal memory footprint.

### Network Topology
`neural_parameters.h`

- **Input Layer:** 4 Neurons (Load, Hour, Day of Week, Month).
- **Hidden Layer:** 8 Neurons (Configurable via macro) with Sigmoid activation.
- **Output Layer:** 1 Neuron (Predicted Load for the next hour - t+1).

---

## Repository Structure

```text
CUDA_Power_Grid_Forecasting
├── README.md                              # Project documentation
├── .gitignore                             # Git ignore rules for datasets & binaries
├── makefile                               # Build, data ingestion & simulation automation
├── neural_parameters.h                    # Global parameters & network topology macros
│
├── train_neural.cu                        # Main CUDA execution core (Parallel Training)
├── run_neural.c                           # CPU inference engine
├── make_logs.py                           # Python data pipeline & SCADA API parser
├── Hourly_Power_Demand_Prediction.pdf     # Official benchmark & evaluation report
│
└── admie_links.json                       # Source JSON containing ADMIE API endpoints

```

*(Note: `admie_dataset.txt` and `weights.txt` are dynamically generated during execution and excluded from version control).*

---

## Performance & Convergence Evaluation

Model training efficiency is evaluated using standard execution timing and the convergence rate of the 16,384 parallel networks.

* **Execution Speed:** Training 16,384 distinct networks over 1,000 epochs with a dataset of 9,600 samples takes approximately **~57 seconds** on modern NVIDIA hardware.
* **Convergence Optimization:** The parallel random search approach successfully guarantees convergence by bypassing local minima. The system automatically selects the network topology that falls below the strict `ERROR_THRESHOLD` (MSE: 0.001) fastest.

---

## How to Compile & Run

### Requirements

* NVIDIA GPU with CUDA Toolkit (`nvcc`)
* GNU Compiler Collection (`gcc`)
* GNU Make
* Python 3 (with `pandas`, `requests`, `openpyxl`)

### Commands

```bash
# 1. Dataset Generation :
# - Parses admie_links.json
# - Fetches Excel files from ADMIE API
# - Outputs normalized admie_dataset.txt

make dataset

# 2. GPU Training Execution :
# - Compiles the CUDA kernel
# - Loads 9,600 samples into VRAM
# - Trains 16,384 parallel networks and saves best weights

make train

# 3. CPU Inference Demo :
# - Compiles the C execution core
# - Prompts user for current grid variables
# - Outputs the next-hour load prediction

make run

# Clean build artifacts

make clean

```

### Output example

```bash
# GPU Training Execution
koulofotias@Stathis:~/myprojects/neural$ make train
nvcc -O3 train_neural.cu -o train
./train
Successfully loaded 9600 samples.
Loaded 9600 samples from file.
Launching 16384 Parallel Neural Networks with 9599 samples...

Execution Time: 57.7795 seconds
Network #305 converged in 87 epochs (MSE: 0.0009787555).
Weights saved to 'weights.txt'.

# CPU Inference Execution
koulofotias@Stathis:~/myprojects/neural$ make run
gcc -Wall -g run_neural.c -o run -lm
./run
This trained neural network is designed to predict Greece's power grid load within the next hour.
Data source: ADMIE API, data from 19/07/2025 to 22/08/2026

--- New Prediction ---
Enter : current grid load (in MW)
        current hour (0 - 23)
        current day of week (MO = 0 to SU = 6)
        current month (JAN = 1 to DEC = 12)
 -- FORMAT: [ Load,Hour,DoW,Month ] --
6500, 14, 0, 7

Predicted Load (Next Hour): 6720 MW
6720

More? (y/n):
n

```

---

## License

MIT License — See `LICENSE` file for details.

```

```

