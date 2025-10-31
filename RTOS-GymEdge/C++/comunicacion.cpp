#include <Arduino.h>
#include <Wire.h>
#include "BluetoothSerial.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Preferences.h>
#include <math.h>

// =================== CONFIGURACIÓN ===================
const char* BT_NAME               = "ESP32-SPP";
const unsigned long SAMPLE_PERIOD_MS = 200;    // 5 Hz
const unsigned long PACKET_PERIOD_MS = 10000;  // enviar cada 10 s
// =====================================================

BluetoothSerial SerialBT;
Adafruit_MPU6050 mpu;
Preferences prefs;

// Offsets guardados en flash (calibración)
float accX_offset = 0, accY_offset = 0, accZ_offset = 0; // m/s^2
float gyroX_offset = 0, gyroY_offset = 0, gyroZ_offset = 0; // rad/s

// Buffer de paquete (todas las muestras de 10 s)
String packetBuffer = "";

// Temporización
unsigned long lastSample = 0;
unsigned long lastPacket = 0;
unsigned long t0 = 0;

// ---------- Prototipos ----------
void calibrarMPU6050();
void guardarOffsets();
bool cargarOffsets();

// =====================================================
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
  // Rangos recomendados (ajusta si deseas)
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

  // Cabecera opcional (si quieres ver con terminal)
  // ¡NO la envíes si tu Python no la ignora!
  // SerialBT.println("timestamp_ms,tempC,hum,ax,ay,az");

  t0 = millis();
  lastSample = lastPacket = millis();
}

void loop() {
  unsigned long now = millis();

  // 1) Muestreo cada SAMPLE_PERIOD_MS
  if (now - lastSample >= SAMPLE_PERIOD_MS) {
    lastSample = now;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Aplicar offsets
    float ax_ms2 = a.acceleration.x - accX_offset;  // m/s^2
    float ay_ms2 = a.acceleration.y - accY_offset;
    float az_ms2 = a.acceleration.z - accZ_offset;

    float gx_rs = g.gyro.x - gyroX_offset;          // rad/s
    float gy_rs = g.gyro.y - gyroY_offset;
    float gz_rs = g.gyro.z - gyroZ_offset;

    // Convertir aceleración a "g"
    const float INV_G = 1.0f / 9.81f;
    float ax_g = ax_ms2 * INV_G;
    float ay_g = ay_ms2 * INV_G;
    float az_g = az_ms2 * INV_G;

    // Temperatura del chip (°C)
    float tempC = temp.temperature;

    // "hum" en el CSV: usaremos |gyro| (magnitud) en deg/s para no tocar Python
    // Convertir rad/s a deg/s: 180/pi ≈ 57.2957795
    const float RAD2DEG = 57.2957795f;
    float gyro_mag_dps = sqrtf(gx_rs*gx_rs + gy_rs*gy_rs + gz_rs*gz_rs) * RAD2DEG;

    // Añadir línea al paquete (terminada en ';')
    // Formato: timestamp_ms,tempC,hum,ax,ay,az;
    char line[96];
    snprintf(line, sizeof(line), "%lu,%.2f,%.2f,%.3f,%.3f,%.3f;",
             now, tempC, gyro_mag_dps, ax_g, ay_g, az_g);
    packetBuffer += line;

    // (Opcional debug por USB)
    // Serial.println(line);
  }

  // 2) Enviar paquete cada PACKET_PERIOD_MS
  if (now - lastPacket >= PACKET_PERIOD_MS) {
    lastPacket = now;

    if (packetBuffer.length() > 0) {
      SerialBT.println(packetBuffer);  // Envío SPP (una sola línea larga)
      Serial.println("📤 Paquete enviado (primeros 120 chars):");
      Serial.println(packetBuffer.substring(0, 120) + " ...");
      packetBuffer = "";               // limpiar para el siguiente bloque
    }
  }
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

  accX_offset = accX_sum / N;
  accY_offset = accY_sum / N;
  // Restar gravedad para que en reposo ~ (0,0,1g)
  accZ_offset = (accZ_sum / N) - 9.81f;

  gyroX_offset = gyroX_sum / N;
  gyroY_offset = gyroY_sum / N;
  gyroZ_offset = gyroZ_sum / N;
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
    return true;
  }
  return false;
}
