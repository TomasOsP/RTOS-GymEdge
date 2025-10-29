#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// Variables de offset
float accX_offset = 0, accY_offset = 0, accZ_offset = 0;
float gyroX_offset = 0, gyroY_offset = 0, gyroZ_offset = 0;


// -------------------------------------------------------------
// Función para calibrar el MPU6050
// -------------------------------------------------------------
void calibrarMPU6050() {
  const int N = 1000; // número de lecturas para promedio
  float accX_sum = 0, accY_sum = 0, accZ_sum = 0;
  float gyroX_sum = 0, gyroY_sum = 0, gyroZ_sum = 0;

  Serial.println("Calibrando... Mantén el sensor quieto y nivelado.");
  delay(2000);

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

  // Promedios
  accX_offset = accX_sum / N;
  accY_offset = accY_sum / N;
  // Para el eje Z restamos 9.81 m/s² (gravedad)
  accZ_offset = (accZ_sum / N) - 9.81;

  gyroX_offset = gyroX_sum / N;
  gyroY_offset = gyroY_sum / N;
  gyroZ_offset = gyroZ_sum / N;
}

void setup() {
  Serial.begin(115200);
  while (!Serial)
    delay(10);

  Serial.println("Iniciando MPU6050...");
  if (!mpu.begin()) {
    Serial.println("No se encontró el MPU6050. Revisa las conexiones!");
    while (1) yield();
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(1000);
  calibrarMPU6050();

  Serial.println("\nCalibración completa. Offsets obtenidos:");
  Serial.print("Acelerómetro -> X: "); Serial.print(accX_offset);
  Serial.print("  Y: "); Serial.print(accY_offset);
  Serial.print("  Z: "); Serial.println(accZ_offset);

  Serial.print("Giroscopio -> X: "); Serial.print(gyroX_offset);
  Serial.print("  Y: "); Serial.print(gyroY_offset);
  Serial.print("  Z: "); Serial.println(gyroZ_offset);
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  float ax = a.acceleration.x - accX_offset;
  float ay = a.acceleration.y - accY_offset;
  float az = a.acceleration.z - accZ_offset;

  float gx = g.gyro.x - gyroX_offset;
  float gy = g.gyro.y - gyroY_offset;
  float gz = g.gyro.z - gyroZ_offset;

  Serial.print("Acelerómetro (m/s^2): ");
  Serial.print(ax); Serial.print(", ");
  Serial.print(ay); Serial.print(", ");
  Serial.println(az);

  Serial.print("Giroscopio (rad/s): ");
  Serial.print(gx); Serial.print(", ");
  Serial.print(gy); Serial.print(", ");
  Serial.println(gz);

  Serial.println("-----------------------");
  delay(500);
}
