#include <Arduino.h>
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Preferences.h>

// ------------------------------------------------------------------
// CONFIGURACIÓN MPU6050
// ------------------------------------------------------------------
Adafruit_MPU6050 mpu;
Preferences prefs;

// Offsets globales
float accX_offset = 0, accY_offset = 0, accZ_offset = 0;
float gyroX_offset = 0, gyroY_offset = 0, gyroZ_offset = 0;

// Variables globales para lecturas corregidas
float ax, ay, az;
float gx, gy, gz;

// Handles de tareas
TaskHandle_t TaskCalibracionHandle;
TaskHandle_t TaskLecturaHandle;

// Frecuencia de muestreo deseada (Hz)
const float SAMPLE_RATE_HZ = 50.0;
const int SAMPLE_PERIOD_MS = 1000 / SAMPLE_RATE_HZ;

// ------------------------------------------------------------------
// DECLARACIÓN DE FUNCIONES
// ------------------------------------------------------------------
void TaskCalibracion(void *pvParameters);
void TaskLectura(void *pvParameters);
void calibrarMPU6050();
void guardarOffsets();
bool cargarOffsets();

// ------------------------------------------------------------------
// SETUP
// ------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Wire.begin();

  Serial.println("Inicializando MPU6050...");
  if (!mpu.begin()) {
    Serial.println("No se encontró el MPU6050. Revisa conexiones!");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  prefs.begin("mpu6050", false);  // namespace
  // Crear tareas
  xTaskCreatePinnedToCore(TaskCalibracion, "Calibracion", 4096, NULL, 2, &TaskCalibracionHandle, 1);
  xTaskCreatePinnedToCore(TaskLectura, "Lectura", 4096, NULL, 1, &TaskLecturaHandle, 0);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}

// ------------------------------------------------------------------
// TAREA: CALIBRACIÓN
// ------------------------------------------------------------------
void TaskCalibracion(void *pvParameters) {
  Serial.println("Tarea de calibración iniciada...");

  // Intentar cargar offsets guardados
  if (cargarOffsets()) {
    Serial.println("Offsets cargados desde memoria flash.");
  } else {
    Serial.println("No hay offsets guardados, iniciando calibración...");
    vTaskDelay(pdMS_TO_TICKS(2000));
    calibrarMPU6050();
    guardarOffsets();
    Serial.println("Offsets calibrados y guardados.");
  }

  Serial.print("ACC offsets -> ");
  Serial.print(accX_offset); Serial.print(", ");
  Serial.print(accY_offset); Serial.print(", ");
  Serial.println(accZ_offset);

  Serial.print("GYRO offsets -> ");
  Serial.print(gyroX_offset); Serial.print(", ");
  Serial.print(gyroY_offset); Serial.print(", ");
  Serial.println(gyroZ_offset);

  vTaskSuspend(NULL); // suspende esta tarea
}

// ------------------------------------------------------------------
// TAREA: LECTURA CONTINUA (50 Hz)
// ------------------------------------------------------------------
void TaskLectura(void *pvParameters) {
  sensors_event_t a, g, temp;

  // Esperar a que haya offsets cargados o calibrados
  while (accX_offset == 0 && gyroX_offset == 0) {
    Serial.println("Esperando calibración...");
    vTaskDelay(pdMS_TO_TICKS(1000));
  }

  Serial.println("Tarea de lectura iniciada (50 Hz).");

  TickType_t xLastWakeTime = xTaskGetTickCount();

  for (;;) {
    mpu.getEvent(&a, &g, &temp);

    ax = a.acceleration.x - accX_offset;
    ay = a.acceleration.y - accY_offset;
    az = a.acceleration.z - accZ_offset;

    gx = g.gyro.x - gyroX_offset;
    gy = g.gyro.y - gyroY_offset;
    gz = g.gyro.z - gyroZ_offset;

    Serial.print("ACC (m/s²): ");
    Serial.print(ax); Serial.print(", ");
    Serial.print(ay); Serial.print(", ");
    Serial.println(az);

    Serial.print("GYRO (rad/s): ");
    Serial.print(gx); Serial.print(", ");
    Serial.print(gy); Serial.print(", ");
    Serial.println(gz);
    Serial.println("--------------------");

    // Mantener frecuencia estable de 50 Hz
    vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
  }
}

// ------------------------------------------------------------------
// FUNCIÓN: CALIBRAR MPU6050
// ------------------------------------------------------------------
void calibrarMPU6050() {
  const int N = 1000;
  float accX_sum = 0, accY_sum = 0, accZ_sum = 0;
  float gyroX_sum = 0, gyroY_sum = 0, gyroZ_sum = 0;

  Serial.println("Calibrando... No muevas el sensor.");

  for (int i = 0; i < N; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    accX_sum += a.acceleration.x;
    accY_sum += a.acceleration.y;
    accZ_sum += a.acceleration.z;
    gyroX_sum += g.gyro.x;
    gyroY_sum += g.gyro.y;
    gyroZ_sum += g.gyro.z;

    vTaskDelay(pdMS_TO_TICKS(3));
  }

  accX_offset = accX_sum / N;
  accY_offset = accY_sum / N;
  accZ_offset = (accZ_sum / N) - 9.81; // corregir gravedad

  gyroX_offset = gyroX_sum / N;
  gyroY_offset = gyroY_sum / N;
  gyroZ_offset = gyroZ_sum / N;
}

// ------------------------------------------------------------------
// GUARDAR OFFSETS EN FLASH
// ------------------------------------------------------------------
void guardarOffsets() {
  prefs.putFloat("accX_off", accX_offset);
  prefs.putFloat("accY_off", accY_offset);
  prefs.putFloat("accZ_off", accZ_offset);
  prefs.putFloat("gyroX_off", gyroX_offset);
  prefs.putFloat("gyroY_off", gyroY_offset);
  prefs.putFloat("gyroZ_off", gyroZ_offset);
}

// ------------------------------------------------------------------
// CARGAR OFFSETS DESDE FLASH
// ------------------------------------------------------------------
bool cargarOffsets() {
  if (prefs.isKey("accX_off")) {
    accX_offset = prefs.getFloat("accX_off", 0);
    accY_offset = prefs.getFloat("accY_off", 0);
    accZ_offset = prefs.getFloat("accZ_off", 0);
    gyroX_offset = prefs.getFloat("gyroX_off", 0);
    gyroY_offset = prefs.getFloat("gyroY_off", 0);
    gyroZ_offset = prefs.getFloat("gyroZ_off", 0);
    return true;
  }
  return false;
}
