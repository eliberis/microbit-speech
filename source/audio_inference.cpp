#include "audio_inference.h"
#include "CodalDmesg.h"


AudioInference::AudioInference(const unsigned char* model_data,
                               tflite::ErrorReporter* error_reporter) {
    this->error_reporter = error_reporter;

    this->model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
                "Model provided is schema version %d not equal "
                "to supported version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    InitializeMicroFeatures(this->error_reporter);
}

TfLiteStatus AudioInference::generateFeatures(int16_t *input_data, int8_t *feature_data) {
    for (int s = 0; s < kFeatureSliceCount; s++) {
        const int32_t slice_start_ms = s * kFeatureSliceStrideMs;
        int16_t* audio_samples = &input_data[slice_start_ms * kAudioSampleFrequency / 1000];
        int8_t* new_slice_data = feature_data + (s * kFeatureSliceSize);
        size_t num_samples_read;
        TfLiteStatus generate_status = GenerateMicroFeatures(
            error_reporter, audio_samples, 512, kFeatureSliceSize,
            new_slice_data, &num_samples_read);
        if (generate_status != kTfLiteOk) {
            return generate_status;
        }
    }
    DMESGF("Generated spectrogram features");
    return kTfLiteOk;
}

TfLiteStatus AudioInference::invokeTfLite(const int8_t *input_data, uint8_t *memory_arena, const int tensor_arena_size, int8_t *output_data) {
    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    //
    // static tflite::AllOpsResolver resolver;
    static tflite::MicroMutableOpResolver<6> resolver(error_reporter);
    if (resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk) {
        error_reporter->Report("Operator registration failed.");
        return kTfLiteError;
    }
    
    DMESGF("Creating MicroInterpreter");
    static tflite::MicroInterpreter interpreter(
            this->model, resolver, memory_arena, tensor_arena_size, this->error_reporter);
    DMESGF("MicroInterpreter created");

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
        return allocate_status;
    }
    DMESGF("Model tensors allocated");

    // Get information about the memory area to use for the model's input.
    TfLiteTensor *input = interpreter.input(0);
    if ((input->dims->size != 4) || (input->dims->data[0] != 1) ||
        (input->dims->data[1] != kFeatureSliceCount) ||
        (input->dims->data[2] != kFeatureSliceSize) ||
        (input->type != kTfLiteInt8)) {
        TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
        return kTfLiteError;
    }
    DMESGF("Input shape: (%d, %d, %d, %d)", 
        input->dims->data[0], 
        input->dims->data[1], 
        input->dims->data[2], 
        input->dims->data[3]);

    // Copy over model input
    memcpy(input->data.int8, input_data, kFeatureElementCount);

    // Run the model on the spectrogram input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed");
        return invoke_status;
    }
    DMESGF("Model invoked");

    // Obtain a pointer to the output tensor
    TfLiteTensor* output = interpreter.output(0);

    if ((output->dims->size != 2) ||
        (output->dims->data[0] != 1) ||
        (output->dims->data[1] != kCategoryCount)) {
        error_reporter->Report(
            "The results for recognition should contain %d elements, but there are "
            "%d in an %d-dimensional shape",
            kCategoryCount, output->dims->data[1],
            output->dims->size);
        return kTfLiteError;
    }

    if (output->type != kTfLiteInt8) {
        error_reporter->Report(
            "The results for recognition should be int8 elements, but are %d",
            output->type);
        return kTfLiteError;
    }

    // Copy over model output
    memcpy(output_data, output->data.int8, kCategoryCount);
    DMESGF("Output written and objects deleted.");
    return kTfLiteOk;
}

// int8_t* AudioInference::invoke() {
//     // Convert audio samples into a spectrogram
//     InitializeMicroFeatures(this->error_reporter);

//     for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
//          ++new_slice) {
             
//     size_t num_samples_read;
//     TfLiteStatus generate_status = GenerateMicroFeatures(
//         error_reporter, audio_samples, audio_samples_size, kFeatureSliceSize,
//         new_slice_data, &num_samples_read);
//     if (generate_status != kTfLiteOk) {
//         error_reporter->Report("Feature generation failed");
//         return nullptr;
//     }

    
// }

AudioInference::~AudioInference() {
    delete this->micro_op_resolver;
}