#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include "model.h"
#include <Adafruit_MPU6050.h>
#include <TFT_eSPI.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <FS.h>
#include <SD.h>
#include <SPI.h>
#include <TinyGPS++.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
#include <ArduinoJson.h>
#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

#define SD_CS 10
#define RXD2 19
#define TXD2 20
#define BUTTON1_PIN 15
#define BUTTON2_PIN 16
#define BUTTON3_PIN 17
#define BUTTON4_PIN 18
#define LED1_PIN 4
#define LED2_PIN 5
#define LED3_PIN 6
#define LED4_PIN 7
#define API_KEY       "AIzaSyDAG63UtpyVpp6LfX2yNKfVZZMY-DJ15Nw"
#define DATABASE_URL  "https://rezy-49046-default-rtdb.asia-southeast1.firebasedatabase.app/" 

struct SensorData {
  String timestamp;
  String formattedTime;
  float accX;
  float accY;
  float accZ;
  float gyroX;
  float gyroY;
  float gyroZ;
  double latitude;
  double longitude;
  float hdop;
  int satellites;
  int condition; 
  String description;
};

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

void displayData(String formattedTime);
String floatToString(float value, int decimalPlaces);
String getFormattedTime();
void kirimKeServer(SensorData data);
void logData(String formattedTime);
void simpan(SensorData data);
void startRecording(String filename, int ledPin, uint16_t color, String jalanNama, String jenis, int condition);
void tampilkanStatusWiFi(int x, int y, const char* status);
void turnOffAllLEDs();
String getTimestamp();
String toCSV(SensorData data);
String toJSON(SensorData data);
SensorData collectSensorData();
String getFormattedTime();

constexpr int kTensorArenaSize = 40 * 1024; 
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(8)));

float sensor_buffer[30][6];
int buffer_index = 0;

const char* ssid = "E7";
const char* password = "Kmwaz200";
const long  GMT_OFFSET_SEC     = 7 * 3600;
const int   DST_OFFSET_SEC     = 0;   

Adafruit_MPU6050 mpu;
TFT_eSPI tft = TFT_eSPI();
HardwareSerial neogps(1);
TinyGPSPlus gps;

unsigned long lastSaveTime = 0;
const unsigned long saveInterval = 50;

float alpha = 0.1;
float filteredAccX = 0, filteredAccY = 0, filteredAccZ = 0;
float filteredGyroX = 0, filteredGyroY = 0, filteredGyroZ = 0;

bool isRecording = false;
unsigned long recordStartTime = 0;
String currentFilename = "/data.csv";
String currentJenis = "";
int currentCondition = 0;
uint16_t recordColor = TFT_WHITE;
int iter = 0;
unsigned long last_millis_tft = 0;
const int interval_tft = 700;

unsigned long led_on_time = 0;
const unsigned long led_duration = 1000;
bool led_active = false;
int active_led_pin = -1;

const float mean_values[6] = { 0.01393333, 1.85801667, 10.70826667, -0.09466667, 0.01183333, 0.01583333};
const float std_values[6] = {0.31230198, 0.92441643, 1.56601671, 0.16382375, 0.0606078, 0.08885928};

const char* labels[4] = {"Jalan Normal", "Jalan Berlubang", "Polisi Tidur", "Speed Trap"};
const uint16_t label_colors[4] = {TFT_WHITE, TFT_RED, TFT_CYAN, TFT_YELLOW};
const int led_pins[4] = {LED1_PIN, LED2_PIN, LED3_PIN, LED4_PIN};

void setup() {
  Serial.begin(115200);
  neogps.begin(9600, SERIAL_8N1, RXD2, TXD2);
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(1);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.fillScreen(TFT_BLACK);
  tampilkanStatusWiFi(15, 56, "Connecting to WiFi...");
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");
  tft.fillScreen(TFT_BLACK);
  tampilkanStatusWiFi(35, 50, "WiFi and Server");
  tampilkanStatusWiFi(50, 65, "Connected!");
  delay(2000);
  configTime(GMT_OFFSET_SEC, DST_OFFSET_SEC,
           "pool.ntp.org", "time.nist.gov");
  Serial.print("Menunggu waktu NTP … ");
  struct tm timeinfo;
  while (!getLocalTime(&timeinfo)) {  
    Serial.print(".");
    delay(250);
  }
  Serial.println("OK");
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  if (Firebase.signUp(&config, &auth, "", "")) {Firebase.reconnectWiFi(true);} 
  config.token_status_callback = tokenStatusCallback;
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_YELLOW, TFT_BLACK);
  tft.setCursor(19, 0);
  tft.print("Road Damage Detector");
  pinMode(BUTTON1_PIN, INPUT_PULLUP);
  pinMode(BUTTON2_PIN, INPUT_PULLUP);
  pinMode(BUTTON3_PIN, INPUT_PULLUP);
  pinMode(BUTTON4_PIN, INPUT_PULLUP);
  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);
  pinMode(LED3_PIN, OUTPUT);
  pinMode(LED4_PIN, OUTPUT);
  turnOffAllLEDs();
  if (!mpu.begin(0x68)) {
    Serial.println("MPU6050 sensor initialization failed");
    while (1);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  if (!SD.begin(SD_CS)) {
    Serial.println("SD card initialization failed!");
    while (1);
  }
  Serial.println("SD card was initialized successfully");
  Serial.print("Free heap before model init: ");
  Serial.println(ESP.getFreeHeap());
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema tidak sesuai!");
    while (1);
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  Serial.print("Free heap before AllocateTensors: ");
  Serial.println(ESP.getFreeHeap());
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Alokasi tensor gagal!");
    while (1);
  }
  Serial.print("Free heap after AllocateTensors: ");
  Serial.println(ESP.getFreeHeap());
  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type == kTfLiteFloat32) {
    Serial.println("Input tensor adalah float32");
  } else {
    Serial.println("Input tensor bukan float32!");
    while (1);
  }
  Serial.println("TensorFlow Lite berhasil diinisialisasi");
}

void loop() {
  while (neogps.available()) {
    gps.encode(neogps.read());
  }
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  SensorData currentData = collectSensorData();
  filteredAccX = alpha * a.acceleration.x + (1 - alpha) * filteredAccX;
  filteredAccY = alpha * a.acceleration.y + (1 - alpha) * filteredAccY;
  filteredAccZ = alpha * a.acceleration.z + (1 - alpha) * filteredAccZ;
  filteredGyroX = alpha * g.gyro.x + (1 - alpha) * filteredGyroX;
  filteredGyroY = alpha * g.gyro.y + (1 - alpha) * filteredGyroY;
  filteredGyroZ = alpha * g.gyro.z + (1 - alpha) * filteredGyroZ;
  if (buffer_index < 30) {
    sensor_buffer[buffer_index][0] = filteredAccX;
    sensor_buffer[buffer_index][1] = filteredAccY;
    sensor_buffer[buffer_index][2] = filteredAccZ;
    sensor_buffer[buffer_index][3] = filteredGyroX;
    sensor_buffer[buffer_index][4] = filteredGyroY;
    sensor_buffer[buffer_index][5] = filteredGyroZ;
    buffer_index++;
  }
  if (buffer_index == 30) {
    for (int i = 0; i < 30; i++) {
      for (int j = 0; j < 6; j++) {
        sensor_buffer[i][j] = (sensor_buffer[i][j] - mean_values[j]) / std_values[j];
      }
    }
    memcpy(input->data.f, sensor_buffer, 30 * 6 * sizeof(float));
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Inferensi gagal!");
      return;
    }
    float output_data[4];
    for (int i = 0; i < 4; i++) {
      output_data[i] = output->data.f[i];
    }
    float threshold = 0.7;
    int max_idx = 0;
    float max_prob = output_data[0];
    for (int i = 1; i < 4; i++) {
      if (output_data[i] > max_prob) {
        max_prob = output_data[i];
        max_idx = i;
      }
    }
    if (max_prob > threshold) {
      turnOffAllLEDs();
      digitalWrite(LED2_PIN, HIGH); 
      led_active = true;
      led_on_time = millis();
      active_led_pin = LED2_PIN;
      tft.setTextColor(TFT_RED, TFT_BLACK); 
      tft.fillRect(50, 95, 100, 10, TFT_BLACK);
      tft.setCursor(35, 95);
      tft.print(labels[1]); 
      currentData.condition = 0; 
      currentData.description = "1";
      if (!isRecording) {
        kirimKeServer(currentData);
      }
    }
    buffer_index = 0;
  }
  if (led_active && (millis() - led_on_time >= led_duration)) {
    digitalWrite(active_led_pin, LOW);
    led_active = false;
    active_led_pin = -1;
  }
  if (!isRecording) {
    if (digitalRead(BUTTON1_PIN) == LOW) startRecording("/Jalan_Normal.csv", LED1_PIN, TFT_WHITE, "Jalan Normal", "Jalan Normal", 0);
    if (digitalRead(BUTTON2_PIN) == LOW) startRecording("/Jalan_Berlubang.csv", LED2_PIN, TFT_RED, "Jalan Berlubang", "Jalan Berlubang", 1);
    if (digitalRead(BUTTON3_PIN) == LOW) startRecording("/Polisi_Tidur.csv", LED3_PIN, TFT_CYAN, "Polisi Tidur", "Polisi Tidur", 2);
    if (digitalRead(BUTTON4_PIN) == LOW) startRecording("/Speed_Trap.csv", LED4_PIN, TFT_YELLOW, "Speed Trap", "Speed Trap", 3);
  }
  if (millis() - lastSaveTime >= saveInterval) {
    lastSaveTime = millis();
    if (isRecording) {
      iter++;
      currentData.condition = 0;
      currentData.description = String(currentCondition);
      simpan(currentData); 
      if (iter == 50) {
        kirimKeServer(currentData);
      }
      if (iter >= 100) {
        isRecording = false;
        iter = 0;
        currentFilename = "/data.csv";  
        currentJenis = "";
        currentCondition = 0;
        turnOffAllLEDs();
      }
    } else {
      currentData.condition = 0; 
      currentData.description = "0";
      simpan(currentData); 
    }
    logData(currentData.timestamp);
  }
  if (millis() - last_millis_tft >= interval_tft) {
    last_millis_tft = millis();
    displayData(currentData.formattedTime);
  }
}

SensorData collectSensorData() {
  SensorData data;
  data.timestamp = getTimestamp();
  data.formattedTime = getFormattedTime(); 
  data.accX = filteredAccX;
  data.accY = filteredAccY;
  data.accZ = filteredAccZ;
  data.gyroX = filteredGyroX;
  data.gyroY = filteredGyroY;
  data.gyroZ = filteredGyroZ;
  if (gps.location.isValid()) {
    data.latitude = gps.location.lat();
    data.longitude = gps.location.lng();
  } else {
    data.latitude = 0.0;
    data.longitude = 0.0;
  }
  data.hdop = gps.hdop.hdop();
  data.satellites = gps.satellites.value();
  data.condition = 0; 
  data.description = "0";
  return data;
}

String toCSV(SensorData data) {
  String accXStr = floatToString(data.accX, 1);
  String accYStr = floatToString(data.accY, 1);
  String accZStr = floatToString(data.accZ, 1);
  String gyroXStr = floatToString(data.gyroX, 1);
  String gyroYStr = floatToString(data.gyroY, 1);
  String gyroZStr = floatToString(data.gyroZ, 1);
  String latStr = floatToString(data.latitude, 6);
  String lngStr = floatToString(data.longitude, 6);
  float accuracyValue = data.hdop * 2.5;
  String hdopStr = floatToString(accuracyValue, 2);
  String satsStr = String(data.satellites);
  return data.timestamp + ";" + data.formattedTime + ";" + accXStr + ";" + accYStr + ";" + accZStr + ";" +
         gyroXStr + ";" + gyroYStr + ";" + gyroZStr + ";" +
         latStr + ";" + lngStr + ";" + hdopStr + ";" + satsStr;
}

String toJSON(SensorData data) {
  DynamicJsonDocument doc(1024);
  JsonObject sensor = doc.createNestedObject("sensor");
  JsonObject timeData = sensor.createNestedObject(data.timestamp);
  timeData["longitude"] = String(data.longitude, 6);
  timeData["latitude"] = String(data.latitude, 6);
  float accuracyValue = data.hdop * 2.5;
  timeData["accuracy"] = String(accuracyValue, 2);
  timeData["description"] = data.description;
  timeData["timestamp"] = data.timestamp;
  timeData["clock"] = data.formattedTime;
  timeData["condition"] = data.condition;
  String jsonString;
  serializeJson(doc, jsonString);
  return jsonString;
}

String bacaDataKeN(int n, String filename) {
  File file = SD.open(filename, FILE_READ);
  if (!file) {
    Serial.println("Gagal membuka file untuk dibaca");
    return "";
  }
  int currentLine = 0;
  String line;
  while (file.available() && currentLine < n) {
    line = file.readStringUntil('\n');
    line.trim();
    currentLine++;
  }
  file.close();
  if (currentLine == n && line != "") {
    return line;
  } else {
    Serial.println("Data ke-" + String(n) + " tidak ditemukan");
    return "";
  }
}

void displayData(String formattedTime) { 
  tft.fillRect(0, 15, 160, 100, TFT_BLACK); 
  tft.setTextColor(TFT_CYAN, TFT_BLACK);
  if (gps.location.isValid()) {
      tft.setCursor(5, 20); tft.print("Lat: "); 
      tft.print(gps.location.lat(), 6);
      tft.setCursor(5, 35); tft.print("Long: "); 
      tft.print(gps.location.lng(), 6);
      tft.setCursor(5, 115); tft.print("HDOP: "); 
      tft.print(gps.hdop.hdop(), 2);
      tft.setCursor(110, 115); tft.print("Sat: "); 
      tft.print(gps.satellites.value());
    } else {
      tft.setCursor(5, 20); 
      tft.setTextColor(TFT_RED, TFT_BLACK); 
      tft.print("No GPS Data");
    }
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setCursor(5, 80); 
  tft.print("Time: "); 
  tft.print(formattedTime);
  tft.setTextColor(TFT_CYAN, TFT_BLACK);
  tft.setCursor(5, 50); 
  tft.print("Acc: ");
  tft.print(floatToString(filteredAccX, 1)); 
  tft.print(", ");
  tft.print(floatToString(filteredAccY, 1)); 
  tft.print(", ");
  tft.print(floatToString(filteredAccZ, 1));
  tft.setCursor(5, 65);
  tft.print("Gyro: ");
  tft.print(floatToString(filteredGyroX, 1)); 
  tft.print(", ");
  tft.print(floatToString(filteredGyroY, 1)); 
  tft.print(", ");
  tft.print(floatToString(filteredGyroZ, 1));
}

String floatToString(float value, int decimalPlaces) { 
  char nilai[20]; 
  dtostrf(value, 1, decimalPlaces, nilai); 
  String result = String(nilai); 
  result.replace('.', ','); 
  return result; 
}

void kirimKeServer(SensorData data) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Firebase: WiFi tidak terhubung.");
    return;
  }
  if (!Firebase.ready()) {
    Serial.println("Firebase: Klien belum siap, coba lagi nanti.");
    return;
  }
  String path = "/sensor/" + data.timestamp;
  FirebaseJson payload;
  payload.set("longitude", String(data.longitude, 6));
  payload.set("latitude", String(data.latitude, 6));
  float accuracyValue = data.hdop * 2.5;
  payload.set("accuracy", String(accuracyValue, 2));
  payload.set("condition", data.condition);
  payload.set("timestamp", data.timestamp);
  payload.set("description", data.description);
  Serial.println("-> Mengirim data ke Firebase...");
  Serial.print("   Path: ");
  Serial.println(path);
  if (Firebase.RTDB.setJSON(&fbdo, path.c_str(), &payload)) {
    Serial.println("<- BERHASIL: Data berhasil dikirim ke Firebase.");
    Serial.print("   ETag: ");
    Serial.println(fbdo.ETag()); 
  } else {
    Serial.println("<- GAGAL: Tidak dapat mengirim data.");
    Serial.print("   Alasan: ");
    Serial.println(fbdo.errorReason());
  }
}

void logData(String formattedTime) { 
  Serial.print("Time: "); 
  Serial.print(formattedTime); 
  Serial.print(" | Acc: "); 
  Serial.print(floatToString(filteredAccX, 1)); 
  Serial.print(", "); 
  Serial.print(floatToString(filteredAccY, 1)); 
  Serial.print(", "); 
  Serial.print(floatToString(filteredAccZ, 1)); 
  Serial.print(" | Gyro: "); 
  Serial.print(floatToString(filteredGyroX, 1)); 
  Serial.print(", "); 
  Serial.print(floatToString(filteredGyroY, 1)); 
  Serial.print(", "); 
  Serial.print(floatToString(filteredGyroZ, 1)); 
  Serial.print(" | GPS: "); 
  Serial.print(gps.location.isValid() ? String(gps.location.lat(), 6) + ", " + String(gps.location.lng(), 6) : "No Data"); 
  Serial.print(" | HDOP: "); 
  Serial.print(gps.hdop.hdop(), 2); 
  Serial.print(" | Satelit: "); 
  Serial.println(gps.satellites.value()); 
}

void simpan(SensorData data) { 
  String csvData = toCSV(data);
  File file = SD.open(currentFilename, FILE_APPEND); 
  if (file) { 
    file.println(csvData); 
    file.close(); 
  } else { 
    Serial.println("Gagal membuka file untuk ditulis"); 
  } 
}

void simpanFoto(String filename, uint8_t* data, size_t len) {
  File file = SD.open(filename, FILE_WRITE);
  if (file) {
    file.write(data, len);
    file.close();
    Serial.println("Foto disimpan: " + filename);
  } else {
    Serial.println("Gagal menyimpan foto: " + filename);
  }
}

void startRecording(String filename, int ledPin, uint16_t color, String jalanNama, String jenis, int condition) { 
  if (isRecording) return; 
  isRecording = true; 
  recordStartTime = millis(); 
  lastSaveTime = millis(); 
  iter = 0; 
  turnOffAllLEDs(); 
  digitalWrite(ledPin, HIGH); 
  currentFilename = filename; 
  currentJenis = jenis; 
  currentCondition = condition;
  recordColor = color; 
  tft.setTextColor(color, TFT_BLACK); 
  tft.fillRect(50, 95, 100, 10, TFT_BLACK); 
  tft.setCursor(35, 95); 
  tft.print(jalanNama); 
}

void tampilkanStatusWiFi(int x, int y, const char* status) {
  tft.setTextColor(TFT_CYAN, TFT_BLACK);
  tft.setCursor(x, y);        
  tft.setTextSize(1);
  tft.println(status);        
}

void turnOffAllLEDs() {
  digitalWrite(LED1_PIN, LOW);
  digitalWrite(LED2_PIN, LOW);
  digitalWrite(LED3_PIN, LOW);
  digitalWrite(LED4_PIN, LOW);
}

String getTimestamp() {
  time_t now = time(nullptr);
  struct tm tm_now;
  localtime_r(&now, &tm_now);               
  char buf[30];
  snprintf(buf, sizeof(buf),
           "%04d-%02d-%02dT%02d:%02d:%02d+07:00",
           tm_now.tm_year + 1900,
           tm_now.tm_mon  + 1,
           tm_now.tm_mday,
           tm_now.tm_hour,
           tm_now.tm_min,
           tm_now.tm_sec);
  return String(buf);
}

String getFormattedTime() { 
  unsigned long seconds = millis() / 1000; 
  char nilai[9]; 
  sprintf(nilai, "%02lu:%02lu:%02lu", seconds / 3600, (seconds % 3600) / 60, seconds % 60); 
  return String(nilai); 
}
