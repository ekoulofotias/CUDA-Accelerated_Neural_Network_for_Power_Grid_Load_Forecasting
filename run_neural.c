#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neural_parameters.h"

#define NUM_INPUTS 4
#define NUM_HIDDEN 8
#define NUM_OUTPUTS 1

#define HIDDEN_WEIGHTS (NUM_INPUTS * NUM_HIDDEN) // 32
#define OUTPUT_WEIGHTS (NUM_HIDDEN * NUM_OUTPUTS) // 8

float* sigmoid(float* vector, int num_neurons) {

	int i;

	for (i = 0; i < num_neurons; i++) {
		vector[i] = 1.0 / (1.0 + exp(-vector[i]));
	}
	return vector;
}

perceptron_t* make_perceptron(const float* w, float b, int num_inputs) {

	perceptron_t* perceptron;
	int i;

	perceptron = malloc(sizeof(perceptron_t));
	if (perceptron == NULL) {
		exit (1);
	}

	perceptron->weights = malloc(num_inputs * sizeof(float));
	if (perceptron->weights == NULL) {
		exit (1);
	}

	for (i = 0; i < num_inputs; i++) {
		perceptron->weights[i] = w[i];
	}
	perceptron->bias = b;
	return perceptron;
}

float perceptron_func(perceptron_t* perceptron, const float* input, int num_inputs) {
	
	float result = 0;
	int i;

	for (i = 0; i < num_inputs; i++) {
		result += perceptron->weights[i] * input[i];
	}
	result = result + perceptron->bias;
	return result;
}

float* forward_pass(perceptron_t** layer, const float* input, int num_neurons, int num_inputs) {

	float* outputs;
	int i;

	outputs = malloc(num_neurons * sizeof(float));
	for (i = 0; i < num_neurons; i++) {
		outputs[i] = perceptron_func(layer[i], input, num_inputs);
	}
	outputs = sigmoid(outputs, num_neurons);
	return outputs;
}

void free_layer(perceptron_t** layer, int num_neurons) {

	int i;

	for(i = 0; i < num_neurons; i++) {
		free(layer[i]->weights);
		free(layer[i]);
	}
	free(layer);
}

int load_weights(const char* filename, float* hidden_weights, float* hidden_bias, float* output_weight, float* output_bias) {
	
	int i;

	FILE *file = fopen(filename, "r");
	if (!file) {
		perror("Error opening weights file");
		return 0;
	}

	// Reading hidden weights
	for (i = 0; i < HIDDEN_WEIGHTS; i++) {
		if (fscanf(file, "%f", &hidden_weights[i]) != 1) { fclose(file); return 0; }
	}
	// output_weight hidden biases
	for (i = 0; i < NUM_HIDDEN; i++) {
		if (fscanf(file, "%f", &hidden_bias[i]) != 1) { fclose(file); return 0; }
	}
	// output_weight output weights
	for (i = 0; i < OUTPUT_WEIGHTS; i++) {
		if (fscanf(file, "%f", &output_weight[i]) != 1) { fclose(file); return 0; }
	}
	// output_weight output biases
	for (i = 0; i < NUM_OUTPUTS; i++) {
		if (fscanf(file, "%f", &output_bias[i]) != 1) { fclose(file); return 0; }
	}

	fclose(file);
	return 1;
}

int main(int argc, char* argv[]) {
	int i;
	int weights_loaded;

	float hidden_layer_weights[HIDDEN_WEIGHTS];
	float hidden_layer_bias[NUM_HIDDEN];
	float output_layer_weights[OUTPUT_WEIGHTS];
	float output_layer_bias[NUM_OUTPUTS];

	weights_loaded = load_weights("weights.txt", hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias);
	if (!weights_loaded) {
		printf("Failed to load weights from file. Exiting.\n");
		return 1;
	}

	perceptron_t** hidden_layer;
	perceptron_t** output_layer;
	
	hidden_layer = malloc(NUM_HIDDEN * sizeof(perceptron_t*));
	for (i = 0; i < NUM_HIDDEN; i++) {
		hidden_layer[i] = make_perceptron(&hidden_layer_weights[i * NUM_INPUTS], hidden_layer_bias[i], NUM_INPUTS);
	}

	output_layer = malloc(NUM_OUTPUTS * sizeof(perceptron_t*));
	for (i = 0; i < NUM_OUTPUTS; i++) {
		output_layer[i] = make_perceptron(&output_layer_weights[i * NUM_HIDDEN], output_layer_bias[i], NUM_HIDDEN);
	}

	float* hidden_out;
	float* final_out;
	float user_input[NUM_INPUTS];
	char choice = 'y';

	printf("This trained neural network is designed to predict Greece's power grid load within the next hour.\n");
	printf("Data source: ADMIE API, data from 19/07/2025 to 22/08/2026\n");

	while (choice == 'y' || choice == 'Y') {
	
		printf("\n--- New Prediction ---\n");

		printf("Enter : current grid load (in MW)\n");
		printf("        current hour (0 - 23)\n");
		printf("        current day of week (MO = 0 to SU = 6)\n");
		printf("        current month (JAN = 1 to DEC = 12)\n");
		printf(" -- FORMAT: [ Load,Hour,DoW,Month ] --\n");

		scanf(" %f, %f, %f, %f", &user_input[0], &user_input[1], &user_input[2], &user_input[3]);

		user_input[0] = user_input[0] / 10000.0;
		user_input[1] = user_input[1] / 23.0;
		user_input[2] = (user_input[2] - 1.0) / 6.0;
		user_input[3] = (user_input[3] - 1.0) / 11.0;

		hidden_out = forward_pass(hidden_layer, user_input, NUM_HIDDEN, NUM_INPUTS);
		final_out = forward_pass(output_layer, hidden_out, NUM_OUTPUTS, NUM_HIDDEN);

		printf("\nPredicted Load (Next Hour): %.0f MW\n", final_out[0] * 10000.0);
		printf("%.0f\n", final_out[0] * 10000.0);

		free(hidden_out);
		free(final_out);

		printf("\nMore? (y/n):\n");
		scanf(" %c", &choice);
	}

	free_layer(hidden_layer, NUM_HIDDEN);
	free_layer(output_layer, NUM_OUTPUTS);

	return 0;
}
