/*
 * On-device human activity recognition (SITTING / WALKING / FALLING)
 * ESP32 + MPU6050 (I2C) + TensorFlow Lite Micro, int8 quantized model.
 *
 * AIoT course project - see the repository README for the full pipeline.
 *
 * Sensing pipeline implemented here
 * --------------------------------
 *   MPU6050 @ +-16 g / +-2000 deg/s, DLPF 21 Hz  (anti-aliasing for 50 Hz)
 *   -> I2C burst read every 20 ms                (50 Hz, timer-free scheduler)
 *   -> ring buffer of 128 samples x 6 channels   (2.56 s window)
 *   -> per-channel standardisation (TRAIN stats) -> int8 quantisation
 *   -> TFLite Micro interpreter, inference every 64 new samples (1.28 s)
 *   -> majority-of-3 smoothing + confidence gate -> serial + LEDs
 *
 * The sketch has three phases:
 *   1. SELFTEST  : 9 held-out test windows stored as raw MPU6050 register
 *                  values are pushed through the identical pipeline.
 *   2. BENCHMARK : int8 and float32 versions of the same network are timed
 *                  on device and their arena usage is reported.
 *   3. LIVE      : continuous classification of the simulated sensor.
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#if RUN_MODEL_ZOO
#include "model_zoo.h"
#endif

// ----------------------------------------------------------------- config ---
#define PIN_SDA 21
#define PIN_SCL 22
#define PIN_LED_SITTING 25
#define PIN_LED_WALKING 26
#define PIN_LED_FALLING 27

// All three switches can be overridden from the build system, which is how
// tools/measure_firmware.py produces the flash / RAM numbers in the report:
//   arduino-cli compile --build-property compiler.cpp.extra_flags=-DRUN_BENCHMARK=0
#ifndef RUN_SELFTEST
#define RUN_SELFTEST 1
#endif
#ifndef RUN_BENCHMARK
#define RUN_BENCHMARK 1
#endif
// 0 -> the live classifier uses the int8 model (this is what we deploy)
// 1 -> the live classifier uses the float32 model (used to measure the cost of
//      *not* quantizing: flash, arena and latency)
#ifndef PRIMARY_FLOAT32
#define PRIMARY_FLOAT32 0
#endif
// 1 -> sampling runs in its own FreeRTOS task pinned to core 0 while inference
//      runs on core 1 (the shipped design)
// 0 -> the first iteration: sampling and inference share loop() on one core.
//      Kept buildable so the report can quote the measured cost of that bug.
#ifndef USE_SAMPLING_TASK
#define USE_SAMPLING_TASK 1
#endif
// 1 -> benchmark every (variant, precision) candidate on device and halt. This
//      is how the Pareto frontier in the report is measured rather than guessed.
#ifndef RUN_MODEL_ZOO
#define RUN_MODEL_ZOO 0
#endif
#define BENCHMARK_ITERATIONS 20
#define ZOO_ITERATIONS 10

// Only 5 kernels are needed: the model was deliberately written with explicit
// Conv2D so that no EXPAND_DIMS / RESHAPE kernels have to be linked.
constexpr int kNumOps = 5;

// Arena sizes: started at 8 KiB / 24 KiB, then shrunk to the values the device
// itself reported through arena_used_bytes() (int8 3484 B, float32 8504 B) plus
// a small margin. This is the RAM half of the quantization benefit: the int8
// arena is 2.4x smaller than the float32 one for the identical network.
constexpr int kArenaInt8 = 4 * 1024;      // measured 3484 B
constexpr int kArenaFloat32 = 9 * 1024;   // measured 8504 B

#if PRIMARY_FLOAT32
#define PRIMARY_MODEL kHarModelFloat32
#define PRIMARY_MODEL_LEN kHarModelFloat32_len
#define PRIMARY_ARENA arena_primary
#define PRIMARY_ARENA_SIZE kArenaFloat32
#define PRIMARY_TAG "float32"
#define SECONDARY_MODEL kHarModelInt8
#define SECONDARY_MODEL_LEN kHarModelInt8_len
#define SECONDARY_ARENA_SIZE kArenaInt8
#define SECONDARY_TAG "int8"
#else
#define PRIMARY_MODEL kHarModelInt8
#define PRIMARY_MODEL_LEN kHarModelInt8_len
#define PRIMARY_ARENA arena_primary
#define PRIMARY_ARENA_SIZE kArenaInt8
#define PRIMARY_TAG "int8"
#define SECONDARY_MODEL kHarModelFloat32
#define SECONDARY_MODEL_LEN kHarModelFloat32_len
#define SECONDARY_ARENA_SIZE kArenaFloat32
#define SECONDARY_TAG "float32"
#endif

const float kSamplePeriodUs = 1000000.0f / HAR_SAMPLE_RATE_HZ;  // 20000 us
constexpr float kGravity = 9.80665f;      // Adafruit reports m/s^2
constexpr float kRadToDeg = 57.2957795f;  // Adafruit reports rad/s
constexpr float kConfidenceGate = 0.60f;

// --------------------------------------------------------------- globals ---
Adafruit_MPU6050 mpu;

alignas(16) uint8_t arena_primary[PRIMARY_ARENA_SIZE];
#if RUN_BENCHMARK
alignas(16) uint8_t arena_secondary[SECONDARY_ARENA_SIZE];
tflite::MicroInterpreter* interp_secondary = nullptr;
#endif

tflite::MicroInterpreter* interp = nullptr;
TfLiteTensor* model_in = nullptr;
TfLiteTensor* model_out = nullptr;

// Ring buffer holds physical units (g and deg/s) so it can be printed for
// debugging; standardisation happens when the tensor is filled.
float ring[HAR_WINDOW][HAR_CHANNELS];
uint16_t ring_head = 0;      // next write position
uint32_t samples_total = 0;  // total samples ever written
uint16_t samples_since_inference = 0;

uint8_t recent[3] = {0, 0, 0};
uint8_t recent_n = 0;
uint32_t window_index = 0;

// Sampling timing health. On this device one inference takes ~90 ms, i.e. much
// longer than the 20 ms sample period, so whether sampling and inference share
// a core is not a detail: it decides whether samples are lost.
volatile uint32_t deadline_misses = 0;   // sample ticks that arrived late
volatile uint32_t worst_late_us = 0;     // worst lateness observed
volatile uint32_t samples_skipped = 0;   // sample slots that were never read

#if USE_SAMPLING_TASK
// Producer (core 0) / consumer (core 1) hand-off. The producer snapshots a
// complete window so the consumer never reads a buffer that is being written.
float pending_window[HAR_WINDOW * HAR_CHANNELS];
SemaphoreHandle_t window_ready = nullptr;
SemaphoreHandle_t window_lock = nullptr;
volatile uint32_t windows_dropped = 0;   // consumer too slow for the producer
static void sampling_task(void* unused);
#endif

// ------------------------------------------------------------ interpreter ---
using HarOpResolver = tflite::MicroMutableOpResolver<kNumOps>;

static HarOpResolver& op_resolver() {
  static HarOpResolver resolver;
  static bool ready = false;
  if (!ready) {
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();  // GlobalAveragePooling2D
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    ready = true;
  }
  return resolver;
}

static tflite::MicroInterpreter* make_interpreter(const unsigned char* model_data,
                                                 uint8_t* arena, int arena_size,
                                                 const char* tag) {
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[%s] schema mismatch: model=%lu lib=%d\n", tag,
                  (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return nullptr;
  }
  auto* interpreter =
      new tflite::MicroInterpreter(model, op_resolver(), arena, arena_size);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("[%s] AllocateTensors failed (arena %d B too small)\n", tag,
                  arena_size);
    return nullptr;
  }
  Serial.printf("[%s] arena_used=%u B of %d B allocated\n", tag,
                (unsigned)interpreter->arena_used_bytes(), arena_size);
  return interpreter;
}

// ------------------------------------------------------- tensor filling ----
// Copy a window (physical units, oldest sample first) into an interpreter
// input, applying standardisation and, for int8 models, quantisation.
static void fill_input(TfLiteTensor* input, const float* window) {
  const float scale = input->params.scale;
  const int zero_point = input->params.zero_point;
  const int n = HAR_WINDOW * HAR_CHANNELS;

  if (input->type == kTfLiteInt8) {
    for (int i = 0; i < n; ++i) {
      const int c = i % HAR_CHANNELS;
      const float z = (window[i] - kHarNormMean[c]) / kHarNormStd[c];
      int32_t q = lroundf(z / scale) + zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input->data.int8[i] = (int8_t)q;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      const int c = i % HAR_CHANNELS;
      input->data.f[i] = (window[i] - kHarNormMean[c]) / kHarNormStd[c];
    }
  }
}

static void read_output(TfLiteTensor* output, float* probs) {
  if (output->type == kTfLiteInt8) {
    for (int k = 0; k < HAR_NUM_CLASSES; ++k) {
      probs[k] = (output->data.int8[k] - output->params.zero_point) *
                 output->params.scale;
    }
  } else {
    for (int k = 0; k < HAR_NUM_CLASSES; ++k) probs[k] = output->data.f[k];
  }
}

static uint8_t argmax(const float* v, int n) {
  uint8_t best = 0;
  for (int i = 1; i < n; ++i) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

// Copy the ring buffer into a linear, chronologically ordered window.
static void snapshot_window(float* dst) {
  for (int i = 0; i < HAR_WINDOW; ++i) {
    const int src = (ring_head + i) % HAR_WINDOW;
    memcpy(dst + i * HAR_CHANNELS, ring[src], HAR_CHANNELS * sizeof(float));
  }
}

// ------------------------------------------------------------- self test ---
#if RUN_SELFTEST
static void run_selftest() {
  Serial.println(F("--- SELFTEST: held-out test windows as raw MPU6050 counts ---"));
  static float window[HAR_WINDOW * HAR_CHANNELS];
  int passed = 0;

  for (int w = 0; w < HAR_SELFTEST_COUNT; ++w) {
    const int16_t* raw = &kHarSelfTestRaw[w * HAR_WINDOW * HAR_CHANNELS];
    for (int i = 0; i < HAR_WINDOW * HAR_CHANNELS; ++i) {
      const int c = i % HAR_CHANNELS;
      window[i] = (c < 3) ? raw[i] / HAR_ACCEL_LSB_PER_G
                          : raw[i] / HAR_GYRO_LSB_PER_DPS;
    }
    fill_input(model_in, window);
    if (interp->Invoke() != kTfLiteOk) {
      Serial.println(F("Invoke failed"));
      continue;
    }
    float probs[HAR_NUM_CLASSES];
    read_output(model_out, probs);
    const uint8_t pred = argmax(probs, HAR_NUM_CLASSES);
    const uint8_t truth = (uint8_t)kHarSelfTestLabels[w];
    const bool ok = pred == truth;
    passed += ok ? 1 : 0;
    Serial.printf("  [%d] expected=%-8s predicted=%-8s p=%.3f  %s\n", w,
                  kHarClassNames[truth], kHarClassNames[pred], probs[pred],
                  ok ? "PASS" : "FAIL");
  }
  Serial.printf("SELFTEST %d/%d correct\n\n", passed, HAR_SELFTEST_COUNT);
}
#endif

// ------------------------------------------------------------- benchmark ---
#if RUN_BENCHMARK
static void benchmark_one(tflite::MicroInterpreter* target, const char* tag,
                          uint32_t model_bytes, const float* window) {
  if (target == nullptr) return;
  TfLiteTensor* input = target->input(0);
  static uint32_t samples[BENCHMARK_ITERATIONS];

  fill_input(input, window);
  target->Invoke();  // warm up (first call touches all arena pages)

  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
    fill_input(input, window);
    const uint32_t t0 = micros();
    target->Invoke();
    samples[i] = micros() - t0;
  }

  uint32_t total = 0, mn = 0xFFFFFFFF, mx = 0;
  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
    total += samples[i];
    if (samples[i] < mn) mn = samples[i];
    if (samples[i] > mx) mx = samples[i];
  }
  // insertion sort for the p95
  for (int i = 1; i < BENCHMARK_ITERATIONS; ++i) {
    uint32_t key = samples[i];
    int j = i - 1;
    while (j >= 0 && samples[j] > key) {
      samples[j + 1] = samples[j];
      --j;
    }
    samples[j + 1] = key;
  }
  const uint32_t p95 = samples[(int)(BENCHMARK_ITERATIONS * 0.95f)];
  const float mean = (float)total / BENCHMARK_ITERATIONS;

  Serial.printf(
      "BENCH %-8s model=%5lu B arena=%5u B mean=%7.1f us p95=%6lu us "
      "min=%6lu us max=%6lu us duty=%.2f%%\n",
      tag, (unsigned long)model_bytes, (unsigned)target->arena_used_bytes(),
      mean, (unsigned long)p95, (unsigned long)mn, (unsigned long)mx,
      100.0f * mean / (HAR_STRIDE * kSamplePeriodUs));
}

static void run_benchmark() {
  Serial.println(F("--- BENCHMARK: same network, int8 vs float32, on device ---"));
  static float window[HAR_WINDOW * HAR_CHANNELS];
  const int16_t* raw = kHarSelfTestRaw;
  for (int i = 0; i < HAR_WINDOW * HAR_CHANNELS; ++i) {
    const int c = i % HAR_CHANNELS;
    window[i] = (c < 3) ? raw[i] / HAR_ACCEL_LSB_PER_G
                        : raw[i] / HAR_GYRO_LSB_PER_DPS;
  }
  benchmark_one(interp, PRIMARY_TAG, PRIMARY_MODEL_LEN, window);
  benchmark_one(interp_secondary, SECONDARY_TAG, SECONDARY_MODEL_LEN, window);
  Serial.printf("free heap after init: %lu B\n\n",
                (unsigned long)ESP.getFreeHeap());
}
#endif

// -------------------------------------------------------------- model zoo ---
#if RUN_MODEL_ZOO
// One arena, sized for the largest candidate (large/float32). Each interpreter
// lives only for the duration of its own measurement.
alignas(16) uint8_t arena_zoo[64 * 1024];

static void decode_selftest_window(int index, float* out) {
  const int16_t* raw = &kHarSelfTestRaw[index * HAR_WINDOW * HAR_CHANNELS];
  for (int i = 0; i < HAR_WINDOW * HAR_CHANNELS; ++i) {
    const int c = i % HAR_CHANNELS;
    out[i] = (c < 3) ? raw[i] / HAR_ACCEL_LSB_PER_G : raw[i] / HAR_GYRO_LSB_PER_DPS;
  }
}

static void run_model_zoo() {
  static float windows[HAR_SELFTEST_COUNT][HAR_WINDOW * HAR_CHANNELS];
  for (int w = 0; w < HAR_SELFTEST_COUNT; ++w) decode_selftest_window(w, windows[w]);

  Serial.println(F("--- MODEL ZOO: every candidate measured on this device ---"));
  Serial.println(F("ZOO variant precision bytes arena mean_us p95_us correct"));

  for (int m = 0; m < HAR_ZOO_COUNT; ++m) {
    const HarZooEntry& entry = kHarZoo[m];
    const tflite::Model* model = tflite::GetModel(entry.data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.printf("ZOO %s %s SCHEMA_MISMATCH\n", entry.variant, entry.precision);
      continue;
    }
    tflite::MicroInterpreter interpreter(model, op_resolver(), arena_zoo,
                                         sizeof(arena_zoo));
    if (interpreter.AllocateTensors() != kTfLiteOk) {
      Serial.printf("ZOO %s %s ARENA_TOO_SMALL\n", entry.variant, entry.precision);
      continue;
    }
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // accuracy on the embedded held-out windows, so the report can show that
    // the device reproduces the host predictions for every candidate
    int correct = 0;
    for (int w = 0; w < HAR_SELFTEST_COUNT; ++w) {
      fill_input(input, windows[w]);
      if (interpreter.Invoke() != kTfLiteOk) continue;
      float probs[HAR_NUM_CLASSES];
      read_output(output, probs);
      if (argmax(probs, HAR_NUM_CLASSES) == (uint8_t)kHarSelfTestLabels[w]) ++correct;
    }

    uint32_t samples[ZOO_ITERATIONS];
    fill_input(input, windows[0]);
    interpreter.Invoke();  // warm up
    for (int i = 0; i < ZOO_ITERATIONS; ++i) {
      fill_input(input, windows[0]);
      const uint32_t t0 = micros();
      interpreter.Invoke();
      samples[i] = micros() - t0;
    }
    uint32_t total = 0;
    for (int i = 1; i < ZOO_ITERATIONS; ++i) {
      const uint32_t key = samples[i];
      int j = i - 1;
      while (j >= 0 && samples[j] > key) { samples[j + 1] = samples[j]; --j; }
      samples[j + 1] = key;
    }
    for (int i = 0; i < ZOO_ITERATIONS; ++i) total += samples[i];

    Serial.printf("ZOO %s %s %u %u %.1f %lu %d/%d\n", entry.variant,
                  entry.precision, entry.length,
                  (unsigned)interpreter.arena_used_bytes(),
                  (float)total / ZOO_ITERATIONS,
                  (unsigned long)samples[(int)(ZOO_ITERATIONS * 0.9f)], correct,
                  HAR_SELFTEST_COUNT);
  }
  Serial.println(F("ZOO DONE"));
}
#endif

// ------------------------------------------------------------------ setup ---
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println(F("=== ESP32 on-device HAR (SITTING / WALKING / FALLING) ==="));
  Serial.printf("variant=%s window=%d @ %d Hz stride=%d\n", HAR_MODEL_VARIANT,
                HAR_WINDOW, HAR_SAMPLE_RATE_HZ, HAR_STRIDE);

  pinMode(PIN_LED_SITTING, OUTPUT);
  pinMode(PIN_LED_WALKING, OUTPUT);
  pinMode(PIN_LED_FALLING, OUTPUT);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);  // fast mode: a 14 byte burst costs ~0.4 ms
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println(F("MPU6050 not found - check wiring"));
    while (true) delay(1000);
  }
  // Ranges match the ones the training data was clipped to, so the deployed
  // sensor can represent every value the model was trained on.
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
  // 21 Hz DLPF < Nyquist (25 Hz) -> anti-aliasing for the 50 Hz sampling.
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println(F("MPU6050 ready: +-16 g, +-2000 deg/s, DLPF 21 Hz"));

#if RUN_MODEL_ZOO
  run_model_zoo();
  Serial.println(F("READY"));
  return;  // measurement build: no live classification
#endif

  interp = make_interpreter(PRIMARY_MODEL, PRIMARY_ARENA, PRIMARY_ARENA_SIZE,
                            PRIMARY_TAG);
  if (interp == nullptr) {
    while (true) delay(1000);
  }
  model_in = interp->input(0);
  model_out = interp->output(0);
  Serial.printf("live model=%s input scale=%.8f zero_point=%d\n", PRIMARY_TAG,
                model_in->params.scale, model_in->params.zero_point);

#if RUN_BENCHMARK
  interp_secondary = make_interpreter(SECONDARY_MODEL, arena_secondary,
                                      SECONDARY_ARENA_SIZE, SECONDARY_TAG);
#endif

#if RUN_SELFTEST
  run_selftest();
#endif
#if RUN_BENCHMARK
  run_benchmark();
#endif

  // The secondary interpreter exists only for the benchmark above; the live
  // path below uses the primary (by default quantized) model.
#if USE_SAMPLING_TASK
  window_ready = xSemaphoreCreateBinary();
  window_lock = xSemaphoreCreateMutex();
  if (window_ready == nullptr || window_lock == nullptr) {
    Serial.println(F("failed to create synchronisation objects"));
    while (true) delay(1000);
  }
  // Priority above the Arduino loop task (1) so acquisition always wins.
  xTaskCreatePinnedToCore(sampling_task, "sampler", 4096, nullptr, 3, nullptr, 0);
  Serial.println(F("--- LIVE: sampler on core 0, inference on core 1 ---"));
#else
  Serial.println(F("--- LIVE: sampling and inference share core 1 ---"));
#endif
  Serial.println(F("READY"));
}

// ------------------------------------------------------- classification ----
static void classify(const float* window) {
  fill_input(model_in, window);

  const uint32_t t0 = micros();
  const TfLiteStatus status = interp->Invoke();
  const uint32_t dt = micros() - t0;
  if (status != kTfLiteOk) {
    Serial.println(F("Invoke failed"));
    return;
  }

  float probs[HAR_NUM_CLASSES];
  read_output(model_out, probs);
  const uint8_t pred = argmax(probs, HAR_NUM_CLASSES);

  // Temporal smoothing: majority of the last three windows, and a confidence
  // gate so a single low-confidence window cannot raise a fall alarm.
  recent[window_index % 3] = pred;
  if (recent_n < 3) ++recent_n;
  uint8_t votes[HAR_NUM_CLASSES] = {0};
  for (uint8_t i = 0; i < recent_n; ++i) votes[recent[i]]++;
  uint8_t smoothed = pred;
  for (uint8_t k = 0; k < HAR_NUM_CLASSES; ++k) {
    if (votes[k] > votes[smoothed]) smoothed = k;
  }
  const bool confident = probs[pred] >= kConfidenceGate;

  digitalWrite(PIN_LED_SITTING, smoothed == 0 && confident);
  digitalWrite(PIN_LED_WALKING, smoothed == 1 && confident);
  digitalWrite(PIN_LED_FALLING, smoothed == 2 && confident);

  Serial.printf(
      "PRED %-8s p=%.3f [%.2f %.2f %.2f] smoothed=%-8s t=%lu us w=%lu "
      "late=%lu worst=%lu us skipped=%lu\n",
      kHarClassNames[pred], probs[pred], probs[0], probs[1], probs[2],
      kHarClassNames[smoothed], (unsigned long)dt, (unsigned long)window_index,
      (unsigned long)deadline_misses, (unsigned long)worst_late_us,
      (unsigned long)samples_skipped);
  if (smoothed == 2 && confident) Serial.println(F("ALERT fall detected"));
  ++window_index;
}

// Read one sample into the ring buffer. Returns true when a new window is due.
static bool acquire_sample() {
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  float* slot = ring[ring_head];
  slot[0] = accel.acceleration.x / kGravity;  // m/s^2 -> g
  slot[1] = accel.acceleration.y / kGravity;
  slot[2] = accel.acceleration.z / kGravity;
  slot[3] = gyro.gyro.x * kRadToDeg;  // rad/s -> deg/s
  slot[4] = gyro.gyro.y * kRadToDeg;
  slot[5] = gyro.gyro.z * kRadToDeg;

  ring_head = (ring_head + 1) % HAR_WINDOW;
  ++samples_total;
  ++samples_since_inference;

  // Wait for a full window, then infer every HAR_STRIDE new samples (50 %
  // overlap), i.e. inference is duty cycled to 1 of every 64 sample ticks.
  if (samples_total >= HAR_WINDOW && samples_since_inference >= HAR_STRIDE) {
    samples_since_inference = 0;
    return true;
  }
  return false;
}

// --------------------------------------------------------------- schedule ---
// Track how late each sample tick was. `period_us` is the nominal 20 000 us.
static void account_for_lateness(uint32_t late_us) {
  if (late_us == 0) return;
  ++deadline_misses;
  if (late_us > worst_late_us) worst_late_us = late_us;
  samples_skipped += late_us / (uint32_t)kSamplePeriodUs;
}

#if RUN_MODEL_ZOO
void loop() { delay(1000); }  // measurement build: everything happens in setup()

#elif USE_SAMPLING_TASK
// Core 0: nothing but sensor acquisition, so a ~90 ms inference on core 1 can
// never make the 50 Hz stream miss a sample.
static void sampling_task(void* unused) {
  (void)unused;
  TickType_t last_wake = xTaskGetTickCount();
  const TickType_t period_ticks = pdMS_TO_TICKS(1000 / HAR_SAMPLE_RATE_HZ);
  uint32_t expected_us = micros();

  for (;;) {
    const uint32_t now = micros();
    account_for_lateness(now - expected_us > (uint32_t)kSamplePeriodUs
                             ? now - expected_us - (uint32_t)kSamplePeriodUs
                             : 0);
    expected_us = now;

    const bool window_due = acquire_sample();
    if (window_due) {
      if (xSemaphoreTake(window_lock, 0) == pdTRUE) {
        snapshot_window(pending_window);
        xSemaphoreGive(window_lock);
        xSemaphoreGive(window_ready);
      } else {
        ++windows_dropped;  // consumer still busy with the previous window
      }
    }
    vTaskDelayUntil(&last_wake, period_ticks);
  }
}

void loop() {
  static float work[HAR_WINDOW * HAR_CHANNELS];
  if (xSemaphoreTake(window_ready, portMAX_DELAY) != pdTRUE) return;
  xSemaphoreTake(window_lock, portMAX_DELAY);
  memcpy(work, pending_window, sizeof(work));
  xSemaphoreGive(window_lock);
  classify(work);
  if (windows_dropped) {
    Serial.printf("windows dropped by the consumer: %lu\n",
                  (unsigned long)windows_dropped);
  }
}

#else  // first iteration: sampling and inference on the same core
void loop() {
  static float work[HAR_WINDOW * HAR_CHANNELS];
  static uint32_t next_sample_us = 0;
  const uint32_t now = micros();
  if (next_sample_us == 0) next_sample_us = now;
  if ((int32_t)(now - next_sample_us) < 0) return;

  account_for_lateness(now - next_sample_us);
  next_sample_us = now + (uint32_t)kSamplePeriodUs;

  if (acquire_sample()) {
    // Inference blocks the sampling loop for ~90 ms here, which is ~4 sample
    // periods. The `skipped` counter in the PRED line quantifies the loss.
    snapshot_window(work);
    classify(work);
  }
}
#endif
