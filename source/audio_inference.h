#pragma once

#include "micro_features_generator.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


class AudioInference {
private:
	const tflite::Model* model = nullptr;
	tflite::MicroOpResolver *micro_op_resolver = nullptr;
	tflite::ErrorReporter* error_reporter = nullptr;

public:
	AudioInference(const unsigned char* model_data, tflite::ErrorReporter* error_reporter);
	~AudioInference();
	TfLiteStatus invokeTfLite(const int8_t *input_data, uint8_t *memory_arena, const int tensor_arena_size, int8_t *output_data);
	TfLiteStatus generateFeatures(int16_t *input_data, int8_t *feature_data);
};
