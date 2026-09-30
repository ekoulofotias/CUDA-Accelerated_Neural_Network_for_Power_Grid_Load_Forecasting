#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "neural_parameters.h"

__host__ data_sample* read_admie_data(const char* filename, int* out_count) {
	FILE *file = fopen(filename, "r");
	if (!file) {
		perror("Error opening dataset file");
		*out_count = 0;
		return NULL;
	}

	int count = 0;
	char ch;
	while ((ch = fgetc(file)) != EOF) {
		if (ch == '\n') {
			count++;
		}
	}
	rewind(file);

	if (count == 0) {
		fclose(file);
		*out_count = 0;
		return NULL;
	}

	data_sample *samples = (data_sample*)malloc(count * sizeof(data_sample));
	if (!samples) {
		perror("Memory allocation failed");
		fclose(file);
		*out_count = 0;
		return NULL;
	}

	int i = 0;
	while (i < count && fscanf(file, "%f,%f,%f,%f", &samples[i].load, &samples[i].hour, &samples[i].day_of_week, &samples[i].month) == 4) {
		i++;
	}

	fclose(file);
	*out_count = i;
	
	printf("Successfully loaded %d samples.\n", i);

	return samples;
}

__device__ float gpu_rand(unsigned int *seed) {
	*seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
	return ((float)*seed / (float)0x7fffffff) * 2.0 - 1.0;
}

__device__ void sigmoid(float* vector, int num_neurons) {
	int i;
	for (i = 0; i < num_neurons; i++) {
		vector[i] = 1.0 / (1.0 + expf(-vector[i]));
	}
}

__device__ void forward_pass(const float* input, const float* weights, const float* bias, int num_inputs, int num_neurons, float* outputs_out) {
	int i;
	int j;
	float sum;

	for (j = 0; j < num_neurons; j++) {
		sum = bias[j];
		for (i = 0; i < num_inputs; i++) {
			sum = sum + input[i] * weights[j * num_inputs + i];
		}
		outputs_out[j] = sum;
	}
	sigmoid(outputs_out, num_neurons);
}

__device__ void local_error(const float* error, const float* output, int num_neurons, float* delta_out) {
	int i;
	for (i = 0; i < num_neurons; i++) {
		delta_out[i] = error[i] * output[i] * (1.0 - output[i]);
	}
}

__device__ void hidden_error_calc(const float* output_weights, const float* output_delta, const float* hidden_outputs, int num_hidden, int num_outputs, float* hidden_delta_out) {
	int i;
	int j;
	float error_sum;
	
	for (i = 0; i < num_hidden; i++) {
		error_sum = 0.0;
		for (j = 0; j < num_outputs; j++) {
			error_sum = error_sum + output_delta[j] * output_weights[j * num_hidden + i];
		}
		hidden_delta_out[i] = error_sum * hidden_outputs[i] * (1.0 - hidden_outputs[i]);
	}
}

__device__ void error_calc(const float* target, const float* output, int num_neurons, float* error_out) {
	int i;
	for (i = 0; i < num_neurons; i++) {
		error_out[i] = target[i] - output[i];
	}
}

__device__ void new_bias(float* bias, const float* delta, int num_neurons) {
	int i;
	for (i = 0; i < num_neurons; i++) {
		bias[i] = bias[i] + LEARNING_RATE * delta[i];
	}
}

__device__ void new_weights(float* weights, const float* input, const float* delta, int num_neurons, int num_inputs) {
	int i;
	int j;
	
	for (j = 0; j < num_neurons; j++) {
		for (i = 0; i < num_inputs; i++) {
			weights[j * num_inputs + i] = weights[j * num_inputs + i] + LEARNING_RATE * delta[j] * input[i];
		}
	}
}

__device__ void mse_calc(const float* error, int num_neurons, float* mse_out) {
	int i;
	float sum = 0.0;
	for (i = 0; i < num_neurons; i++) {
		sum = sum + (error[i] * error[i]);
	}
	*mse_out = sum / num_neurons;
}

__global__ void train(
	const float* inputs, 
	const float* targets, 
	int* winner_epochs,
	float* winner_mse,
	float* winner_hidden_weights,
	float* winner_hidden_bias,
	float* winner_output_weights,
	float* winner_output_bias,
	unsigned int base_seed,
	int num_samples
) {
	int net_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (net_id >= NUM_NETWORKS) {
		return;
	}

	unsigned int seed = base_seed + net_id * 777;

	// Local Registers using parameters
	float hidden_weights[HIDDEN_WEIGHTS]; 
	float hidden_bias[NUM_HIDDEN];
	float output_weights[OUTPUT_WEIGHTS];
	float output_bias[NUM_OUTPUTS];

	for (int i = 0; i < HIDDEN_WEIGHTS; i++) hidden_weights[i] = gpu_rand(&seed) * 1.5f;
	for (int i = 0; i < NUM_HIDDEN; i++) hidden_bias[i] = gpu_rand(&seed) * 0.5f;
	for (int i = 0; i < OUTPUT_WEIGHTS; i++) output_weights[i] = gpu_rand(&seed) * 1.5f;
	for (int i = 0; i < NUM_OUTPUTS; i++) output_bias[i] = gpu_rand(&seed) * 0.5f;

	float hidden_out[NUM_HIDDEN];
	float final_out[NUM_OUTPUTS];
	float output_error[NUM_OUTPUTS];
	float output_delta[NUM_OUTPUTS];
	float hidden_delta[NUM_HIDDEN];

	int epoch = 0;
	float total_mse = 1.0f;

	while (total_mse > ERROR_THRESHOLD && epoch < MAX_EPOCHS) {

		total_mse = 0.0f;

		for (int sample = 0; sample < num_samples; sample++) {
			const float* in = &inputs[sample * NUM_INPUTS]; 
			const float* target = &targets[sample * NUM_OUTPUTS];

			// 1. Forward Pass
			forward_pass(in, hidden_weights, hidden_bias, NUM_INPUTS, NUM_HIDDEN, hidden_out); 
			forward_pass(hidden_out, output_weights, output_bias, NUM_HIDDEN, NUM_OUTPUTS, final_out);

			// 2. Error Calculation
			error_calc(target, final_out, NUM_OUTPUTS, output_error);
			
			float sample_mse;
			mse_calc(output_error, NUM_OUTPUTS, &sample_mse);
			total_mse += sample_mse;

			// 3. Backward Pass & Deltas
			local_error(output_error, final_out, NUM_OUTPUTS, output_delta);
			hidden_error_calc(output_weights, output_delta, hidden_out, NUM_HIDDEN, NUM_OUTPUTS, hidden_delta);

			// 4. Weight & Bias Updates
			new_bias(output_bias, output_delta, NUM_OUTPUTS);
			new_weights(output_weights, hidden_out, output_delta, NUM_OUTPUTS, NUM_HIDDEN);

			new_bias(hidden_bias, hidden_delta, NUM_HIDDEN);
			new_weights(hidden_weights, in, hidden_delta, NUM_HIDDEN, NUM_INPUTS); 
		}

		total_mse /= (float)num_samples; 
		epoch++;
	}

	// Αποθήκευση των δεδομένων αυτού του δικτύου στις αντίστοιχες θέσεις
	winner_epochs[net_id] = epoch;
	winner_mse[net_id] = total_mse;

	for (int i = 0; i < HIDDEN_WEIGHTS; i++) winner_hidden_weights[net_id * HIDDEN_WEIGHTS + i] = hidden_weights[i];
	for (int i = 0; i < NUM_HIDDEN; i++) winner_hidden_bias[net_id * NUM_HIDDEN + i] = hidden_bias[i];
	for (int i = 0; i < OUTPUT_WEIGHTS; i++) winner_output_weights[net_id * OUTPUT_WEIGHTS + i] = output_weights[i];
	for (int i = 0; i < NUM_OUTPUTS; i++) winner_output_bias[net_id * NUM_OUTPUTS + i] = output_bias[i];
}

int main(int argc, char* argv[]) {

	int total_samples = 0;
	int num_samples;
	data_sample *samples;
	float *host_inputs;
	float *host_targets;
	float *device_inputs;
	float *device_targets;
	float *winner_mse;

	float *winner_hid_weights;
	float *winner_hid_bias;
	float *winner_out_weights;
	float *winner_out_bias;

	int *winner_epochs;
	int i;
	int j;
	
	samples = read_admie_data("admie_dataset.txt", &total_samples);
	if (!samples || total_samples == 0) {
		printf("Failed to load dataset.\n");
		return 1;
	}

	printf("Loaded %d samples from file.\n", total_samples);

	// Use all samples read
	num_samples = total_samples;

	host_inputs = (float*)malloc(num_samples * NUM_INPUTS * sizeof(float));
	host_targets = (float*)malloc(num_samples * sizeof(float));

	// Normalize
	for (i = 0; i < num_samples - 1; i++) {
		host_inputs[i * NUM_INPUTS + 0] = samples[i].load / 10000.0f;
		host_inputs[i * NUM_INPUTS + 1] = samples[i].hour / 23.0f;
		host_inputs[i * NUM_INPUTS + 2] = samples[i].day_of_week / 6.0f;
		host_inputs[i * NUM_INPUTS + 3] = samples[i].month / 12.0f;

		// Target: NEXT load value
		host_targets[i] = samples[i + 1].load / 10000.0f; 
	}
	num_samples = num_samples - 1;

	free(samples);

	cudaMallocManaged(&device_inputs, num_samples * NUM_INPUTS * sizeof(float));
	cudaMallocManaged(&device_targets, num_samples * sizeof(float));
	
	// Allocate memory for all networks
	cudaMallocManaged(&winner_epochs, NUM_NETWORKS * sizeof(int));
	cudaMallocManaged(&winner_mse, NUM_NETWORKS * sizeof(float));
	cudaMallocManaged(&winner_hid_weights, NUM_NETWORKS * HIDDEN_WEIGHTS * sizeof(float)); 
	cudaMallocManaged(&winner_hid_bias, NUM_NETWORKS * NUM_HIDDEN * sizeof(float));
	cudaMallocManaged(&winner_out_weights, NUM_NETWORKS * OUTPUT_WEIGHTS * sizeof(float));
	cudaMallocManaged(&winner_out_bias, NUM_NETWORKS * NUM_OUTPUTS * sizeof(float));

	for(i = 0; i < num_samples * NUM_INPUTS; i++) device_inputs[i] = host_inputs[i];
	for(i = 0; i < num_samples; i++) device_targets[i] = host_targets[i];

	printf("Launching %d Parallel Neural Networks with %d samples...\n", NUM_NETWORKS, num_samples);
	
	clock_t start = clock();

	int threadsPerBlock = 256;
	int blocksPerGrid = (NUM_NETWORKS + threadsPerBlock - 1) / threadsPerBlock;
	unsigned int seed = (unsigned int)time(NULL);

	train<<<blocksPerGrid, threadsPerBlock>>>(
		device_inputs, device_targets, winner_epochs, winner_mse,
		winner_hid_weights, winner_hid_bias, winner_out_weights, winner_out_bias, seed, num_samples
	);

	cudaDeviceSynchronize();

	clock_t end = clock();
	float cpu_time = ((float)(end - start)) / CLOCKS_PER_SEC;

	printf("\nExecution Time: %.4f seconds\n", cpu_time);
	
	// CPU side: Find the network with the smallest MSE
	int best_network = -1;
	float best_mse = 999999.0f; // Αρχικοποίηση με μεγάλη τιμή

	for (j = 0; j < NUM_NETWORKS; j++) {
		if (winner_mse[j] < best_mse) {
			best_mse = winner_mse[j];
			best_network = j;
		}
	}

	// If at least one network got under ERROR_THRESHOLD, we save it
	if (best_network != -1 && best_mse <= ERROR_THRESHOLD) {

		printf("Network #%d converged in %d epochs (MSE: %.10f).\n", best_network, winner_epochs[best_network], best_mse);

		// Storing the weights
		FILE *file = fopen("weights.txt", "w");
		if (file == NULL) {
			printf("Error opening weights.txt for writing.\n");
		} else {
			
			for(j = 0; j < HIDDEN_WEIGHTS; j++) fprintf(file, "%.10f\n", winner_hid_weights[best_network * HIDDEN_WEIGHTS + j]);
			for(j = 0; j < NUM_HIDDEN; j++) fprintf(file, "%.10f\n", winner_hid_bias[best_network * NUM_HIDDEN + j]);
			for(j = 0; j < OUTPUT_WEIGHTS; j++) fprintf(file, "%.10f\n", winner_out_weights[best_network * OUTPUT_WEIGHTS + j]);
			for(j = 0; j < NUM_OUTPUTS; j++) fprintf(file, "%.10f\n", winner_out_bias[best_network * NUM_OUTPUTS + j]);

			fclose(file);
			printf("Weights saved to 'weights.txt'.\n");
		}
	} else {
		printf("No network converged within %d epochs (Best MSE was %.10f).\n", MAX_EPOCHS, best_mse);
	}

	free(host_inputs);
	free(host_targets);
	cudaFree(device_inputs);
	cudaFree(device_targets);
	cudaFree(winner_epochs);
	cudaFree(winner_mse);
	cudaFree(winner_hid_weights);
	cudaFree(winner_hid_bias);
	cudaFree(winner_out_weights);
	cudaFree(winner_out_bias);

	return 0;
}