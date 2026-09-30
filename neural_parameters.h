#ifndef _PARAMETERS
#define _PARAMETERS

#define LEARNING_RATE 0.5
#define ERROR_THRESHOLD 0.001
#define MAX_EPOCHS 1000
#define NUM_NETWORKS 16384

#define NUM_INPUTS 4 // (Load, Hour, DayOfWeek, Month)
#define NUM_HIDDEN 8
#define NUM_OUTPUTS 1

#define HIDDEN_WEIGHTS (NUM_INPUTS * NUM_HIDDEN)
#define OUTPUT_WEIGHTS (NUM_HIDDEN * NUM_OUTPUTS)

typedef struct {
	float load;
	float hour;
	float day_of_week;
	float month;
} data_sample;

typedef struct{
	float* weights;
	float bias;
} perceptron_t;

#endif