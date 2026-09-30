CC = gcc
NVCC = nvcc
CFLAGS = -Wall -g
NVCCFLAGS = -O3
PYTHON = python3

dataset: admie_links.json
	$(PYTHON) make_logs.py

train: train_neural.cu neural_parameters.h
	$(NVCC) $(NVCCFLAGS) train_neural.cu -o train
	./train

run: run_neural.c neural_parameters.h
	$(CC) $(CFLAGS) run_neural.c -o run -lm
	./run

clean:
	rm -f train run weights.txt admie_dataset.txt
