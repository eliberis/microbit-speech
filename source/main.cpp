#include "MicroBit.h"
#include "memory_sink.h"
#include "audio_inference.h"
#include "model_data.h"

#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"

MicroBit uBit;

constexpr int kTensorArenaSize = 80 * 1024;
constexpr int kRecordingBytes = 2 * kAudioSampleFrequency + 10000;
static uint8_t memory_arena[kTensorArenaSize];


void mic_read(uint8_t *output) {
    DMESGF("Reading microphone data");
    uBit.adc.setSamplePeriod(63); // 15873 Hz
    NRF52ADCChannel* microphone = uBit.adc.getChannel(uBit.io.microphone);
    uBit.io.runmic.setDigitalValue(1); 
    uBit.io.runmic.setHighDrive(true); 
    microphone->setGain(7,0);

    uBit.sleep(200);

    static MemorySink sink(microphone->output, output, kRecordingBytes);

    while (!sink.isDone()) {
        uBit.sleep(200);
    }

    microphone->disable();
    uBit.io.runmic.setDigitalValue(0); 
    // uBit.io.runmic.setHighDrive(true); 
    sink.reset();

    DMESGF("Microphone data read");
}

void mic_serial_print(int16_t *mic_data, int length) {
    int CRLF = 0;

    for(int i = 0; i < length; i++) {
        uBit.serial.printf("%x ", mic_data[i]);
        CRLF++;
        if (CRLF == 16){
            uBit.serial.printf("\r\n");
            CRLF = 0;
        }
    }
    
    if (CRLF > 0) {
        uBit.serial.printf("\r\n");
    }
}

void debug_log_cb(const char* s) {
    DMESGF(s);
}

void preprocess_audio(int16_t *input_data, const int length) {
    for (int i = 0; i < length; i++) {
        input_data[i] = (input_data[i] - 6430) * 8;
    }
}

void tflite_inference() {
    RegisterDebugLogCallback(debug_log_cb);
    static tflite::MicroErrorReporter error_reporter;

    static AudioInference engine(g_model_data, &error_reporter);

    uint8_t *mic_data_addr = &memory_arena[kFeatureElementCount];
    mic_read(mic_data_addr);

    int16_t *input_data = reinterpret_cast<int16_t*>(mic_data_addr);
    preprocess_audio(input_data, kRecordingBytes / 2);

    uint8_t *feature_data_addr = &memory_arena[0];
    int8_t* feature_data = reinterpret_cast<int8_t*>(feature_data_addr);
    engine.generateFeatures(input_data, feature_data);

    uint8_t *output_data_addr = &memory_arena[kFeatureElementCount];
    int8_t* output_data = reinterpret_cast<int8_t*>(output_data_addr);
    engine.invokeTfLite(feature_data, &memory_arena[kFeatureElementCount + kCategoryCount], 
                        kTensorArenaSize - kFeatureElementCount - kCategoryCount, output_data);

    // Print out all scores and find the maximum-scoring label
    int current_top_index = 0;
    int8_t current_top_score = 0;
    for (int i = 0; i < kCategoryCount; ++i) {
        int8_t score = output_data[i];
        uBit.serial.printf("%s: %d\r\n", kCategoryLabels[i], score);
        if (score > current_top_score) {
            current_top_score = score;
            current_top_index = i;
        }
    }

    uBit.serial.printf("Heard: %s\r\n", kCategoryLabels[current_top_index]);
    uBit.display.scroll(kCategoryLabels[current_top_index]);
}

void mic_record_test() {
    uint8_t *mic_data_addr = &memory_arena[0];
    mic_read(mic_data_addr);

    int16_t *input_data = reinterpret_cast<int16_t*>(mic_data_addr);
    preprocess_audio(input_data, kRecordingBytes / 2);
    mic_serial_print(input_data, kRecordingBytes / 2);
    release_fiber();
}

int main()
{
    uBit.init();
    tflite::InitializeTarget();
    // create_fiber(mic_record_test);
    create_fiber(tflite_inference);
    release_fiber();
}
