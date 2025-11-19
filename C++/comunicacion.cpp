
#include <Arduino.h>
#include <Wire.h>
#include "BluetoothSerial.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Preferences.h>
#include <math.h>

// =================== CONFIGURACIÓN ===================
static const char* BT_NAME               = "ESP32-SPP";
static const TickType_t SAMPLE_PERIOD_MS = 200;     // 5 Hz
static const TickType_t PACKET_PERIOD_MS = 200;   // enviar cada 10 s

// Filtro IIR para estimación del vector de gravedad g (LPF):
// tau ~ constante de tiempo del LPF (ajustable). Con 0.8s y dt=0.2s -> alpha ~ 0.8/(0.8+0.2)=0.8
static const float GRAV_TAU_SEC          = 0.8f;

// Cola FreeRTOS: cuántas líneas preformateadas retenemos
static const uint16_t QUEUE_LENGTH       = 128;

// Tamaño de cada línea formateada (incluye ';' y '\0')
static const size_t   LINE_BUF_MAX       = 96;
// =====================================================

BluetoothSerial SerialBT;
Adafruit_MPU6050 mpu;
Preferences prefs;

// Offsets guardados en flash (calibración)
float accX_offset = 0, accY_offset = 0, accZ_offset = 0;    // m/s^2 (promedio en reposo)
float gyroX_offset = 0, gyroY_offset = 0, gyroZ_offset = 0; // rad/s (promedio en reposo)

// Estimación dinámica del vector de gravedad (m/s^2)
volatile float g_est_x = 0.0f, g_est_y = 0.0f, g_est_z = 9.81f; // inicia apuntando a Z

// Cola entre tareas: cada item es una línea ya formateada
typedef struct {
  char text[LINE_BUF_MAX];
} sample_line_t;

static QueueHandle_t qLines = nullptr;

// Tareas
static TaskHandle_t hTaskAcq = nullptr;
static TaskHandle_t hTaskBT  = nullptr;

// Temporización
static uint32_t t0_ms = 0;

// ---------- Prototipos ----------
void calibrarMPU6050();
void guardarOffsets();
bool cargarOffsets();
void task_acq(void* arg);
void task_bt(void* arg);

// =================== SETUP ===================
void setup() {
  Serial.begin(115200);
  delay(300);

  // I2C
  Wire.begin();

  // MPU
  Serial.println("Inicializando MPU6050...");
  if (!mpu.begin()) {
    Serial.println("❌ No se encontró el MPU6050. Revisa conexiones!");
    while (1) delay(1000);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);      // ±2 g
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);           // ±250 deg/s
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);        // antialias básico

  // Preferencias (flash)
  prefs.begin("mpu6050", false);

  // Cargar offsets; si no existen, calibrar
  if (cargarOffsets()) {
    Serial.println("✅ Offsets cargados de flash.");
  } else {
    Serial.println("ℹ️ No hay offsets. Calibrando (no muevas el sensor)...");
    delay(1500);
    calibrarMPU6050();
    guardarOffsets();
    Serial.println("✅ Offsets calibrados y guardados.");
  }

  Serial.printf("ACC offsets (m/s^2):  X=%.4f Y=%.4f Z=%.4f\n", accX_offset, accY_offset, accZ_offset);
  Serial.printf("GYRO offsets (rad/s): X=%.6f Y=%.6f Z=%.6f\n", gyroX_offset, gyroY_offset, gyroZ_offset);

  // Bluetooth SPP
  if (!SerialBT.begin(BT_NAME)) {
    Serial.println("❌ Bluetooth SPP no inició");
    while (1) delay(1000);
  }
  Serial.println(String("✅ BT SPP listo: ") + BT_NAME);

  // Cola
  qLines = xQueueCreate(QUEUE_LENGTH, sizeof(sample_line_t));
  if (!qLines) {
    Serial.println("❌ Falló la creación de la cola");
    while (1) delay(1000);
  }

  // Tiempo base
  t0_ms = millis();

  // Info de protocolo
  const uint32_t samples_per_packet = (uint32_t)(PACKET_PERIOD_MS / SAMPLE_PERIOD_MS);
  Serial.printf("📦 Muestras por paquete: %lu (%.2f Hz × %.2f s)\n",
                (unsigned long)samples_per_packet,
                1000.0f / (float)SAMPLE_PERIOD_MS,
                (float)PACKET_PERIOD_MS / 1000.0f);
  Serial.printf("⏱️ Envío cada: %.2f s (%.2f Hz)\n",
                (float)PACKET_PERIOD_MS / 1000.0f,
                1000.0f / (float)PACKET_PERIOD_MS);

  // Crear tareas (acq en Core 1, BT en Core 0)
  BaseType_t ok1 = xTaskCreatePinnedToCore(task_acq, "acq", 4096, nullptr, 2, &hTaskAcq, 1);
  BaseType_t ok2 = xTaskCreatePinnedToCore(task_bt,  "bt",  4096, nullptr, 1, &hTaskBT,  0);
  if (ok1 != pdPASS || ok2 != pdPASS) {
    Serial.println("❌ No se pudieron crear las tareas");
    while (1) delay(1000);
  }
}

void loop() {
  // No hacemos nada en loop; FreeRTOS maneja las tareas
  vTaskDelay(pdMS_TO_TICKS(1000));
}

// ================== Calibración & Offsets ==================
void calibrarMPU6050() {
  const int N = 1000;
  float accX_sum = 0, accY_sum = 0, accZ_sum = 0;
  float gyroX_sum = 0, gyroY_sum = 0, gyroZ_sum = 0;

  Serial.println("Calibrando... Mantén el sensor QUIETO.");

  for (int i = 0; i < N; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    accX_sum += a.acceleration.x;
    accY_sum += a.acceleration.y;
    accZ_sum += a.acceleration.z;

    gyroX_sum += g.gyro.x;
    gyroY_sum += g.gyro.y;
    gyroZ_sum += g.gyro.z;

    delay(3);
  }

  // Promedios (m/s^2 y rad/s)
  accX_offset = accX_sum / N;
  accY_offset = accY_sum / N;
  accZ_offset = accZ_sum / N;

  gyroX_offset = gyroX_sum / N;
  gyroY_offset = gyroY_sum / N;
  gyroZ_offset = gyroZ_sum / N;

  // Inicializa estimación de gravedad con el promedio
  g_est_x = accX_offset;
  g_est_y = accY_offset;
  g_est_z = accZ_offset;
}

void guardarOffsets() {
  prefs.putFloat("accX_off", accX_offset);
  prefs.putFloat("accY_off", accY_offset);
  prefs.putFloat("accZ_off", accZ_offset);
  prefs.putFloat("gyroX_off", gyroX_offset);
  prefs.putFloat("gyroY_off", gyroY_offset);
  prefs.putFloat("gyroZ_off", gyroZ_offset);
}

bool cargarOffsets() {
  if (prefs.isKey("accX_off")) {
    accX_offset = prefs.getFloat("accX_off", 0);
    accY_offset = prefs.getFloat("accY_off", 0);
    accZ_offset = prefs.getFloat("accZ_off", 0);
    gyroX_offset = prefs.getFloat("gyroX_off", 0);
    gyroY_offset = prefs.getFloat("gyroY_off", 0);
    gyroZ_offset = prefs.getFloat("gyroZ_off", 0);
    // Arranque de g_est con algo razonable
    g_est_x = accX_offset;
    g_est_y = accY_offset;
    g_est_z = accZ_offset;
    return true;
  }
  return false;
}

// =================== TAREA: ADQUISICIÓN ===================
void task_acq(void* arg) {
  const float dt_sec = (float)SAMPLE_PERIOD_MS / 1000.0f;
  const float alpha  = GRAV_TAU_SEC / (GRAV_TAU_SEC + dt_sec);
  const float INV_G  = 1.0f / 9.81f;       // m/s^2 -> g
  const float RAD2DEG = 57.2957795f;

  TickType_t lastWake = xTaskGetTickCount();

  for (;;) {
    // Muestrear con periodo fijo
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));

    // Leer sensores
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Quitar offsets "duros" (bias) del sensor
    float ax_ms2 = a.acceleration.x - accX_offset;  // m/s^2
    float ay_ms2 = a.acceleration.y - accY_offset;
    float az_ms2 = a.acceleration.z - accZ_offset;

    float gx_rs = g.gyro.x - gyroX_offset;          // rad/s
    float gy_rs = g.gyro.y - gyroY_offset;
    float gz_rs = g.gyro.z - gyroZ_offset;

    // Estimar gravedad (LPF) y restarla -> aceleración lineal
    // g_est(k) = alpha*g_est(k-1) + (1-alpha)*acc(k)
    g_est_x = alpha * g_est_x + (1.0f - alpha) * ax_ms2;
    g_est_y = alpha * g_est_y + (1.0f - alpha) * ay_ms2;
    g_est_z = alpha * g_est_z + (1.0f - alpha) * az_ms2;

    float lin_ax_ms2 = ax_ms2 - g_est_x;
    float lin_ay_ms2 = ay_ms2 - g_est_y;
    float lin_az_ms2 = az_ms2 - g_est_z;

    // Convertir aceleración lineal a "g"
    float ax_g = lin_ax_ms2 * INV_G;
    float ay_g = lin_ay_ms2 * INV_G;
    float az_g = lin_az_ms2 * INV_G;

    // Giros: a deg/s (aceleraciones angulares)
    float gx_dps = gx_rs * RAD2DEG;
    float gy_dps = gy_rs * RAD2DEG;
    float gz_dps = gz_rs * RAD2DEG;

    // Timestamp
    uint32_t now_ms = millis();

    // Formatear línea
    sample_line_t line{};
    // NUEVO FORMATO: timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps;
    snprintf(line.text, LINE_BUF_MAX, "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f;",
             (unsigned long)now_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps);

    // Encolar (si la cola está llena, descarta la muestra más vieja para no atrasar)
    if (xQueueSend(qLines, &line, 0) != pdTRUE) {
      sample_line_t dummy;
      xQueueReceive(qLines, &dummy, 0);
      xQueueSend(qLines, &line, 0);
    }
  }
}

// =================== TAREA: BLUETOOTH SPP ===================
void task_bt(void* arg) {
  String packetBuffer; packetBuffer.reserve(4096); // para reducir realocaciones
  TickType_t lastPacketTick = xTaskGetTickCount();

  for (;;) {
    // Espera una línea hasta 50 ms para no bloquear
    sample_line_t line;
    if (xQueueReceive(qLines, &line, pdMS_TO_TICKS(50)) == pdTRUE) {
      packetBuffer += line.text;  // acumular líneas
    }

    // ¿Tiempo de enviar?
    TickType_t now = xTaskGetTickCount();
    if ((now - lastPacketTick) >= pdMS_TO_TICKS(PACKET_PERIOD_MS)) {
      lastPacketTick = now;

      if (packetBuffer.length() > 0) {
        // Envío SPP (una sola línea larga). Mantiene el protocolo de "una línea por paquete".
        SerialBT.println(packetBuffer);
        Serial.print("📤 Paquete enviado (primeros 120 chars): ");
        Serial.println(packetBuffer.substring(0, 120) + " ...");
        packetBuffer = "";
      }
    }
  }
}