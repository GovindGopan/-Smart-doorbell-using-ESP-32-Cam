    /*
 * ==========================================
 * ESP32-CAM AI-THINKER SECURITY SYSTEM
 * ==========================================
 * 
 * DIAGNOSTIC VERSION - With Server Logging
 * 
 * ==========================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>

// ==========================================
// WIFI CONFIGURATION
// ==========================================
const char* ssid = "Wifi name";
const char* password = "Wifi password";

// ==========================================
// SERVER CONFIGURATION
// ==========================================
const char* serverIP = "192.168.1.9";
const char* serverPort = "8000";
const char* uploadPath = "/upload";
const char* logPath = "/buzzer-log";
const char* ipRegistrationPath = "/setup-esp32-ip";
const char* diagnosticLogPath = "/esp32-log";  // NEW: Diagnostic log endpoint

// ==========================================
// ESP32-CAM AI-THINKER PIN DEFINITIONS
// ==========================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM       5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==========================================
// PERIPHERAL PIN DEFINITIONSa
// ==========================================
const int buzzerPin = 12;
const int irSensorPin = 13;
const int buttonPin = 14;

// ==========================================
// BUZZER TIMING CONFIGURATION
// ==========================================
bool buzzerOn = false;
unsigned long buzzerOnTime = 0;
const unsigned long buzzerDuration = 500;

// ==========================================
// PIR SENSOR CONFIGURATION
// ==========================================
unsigned long lastPirTriggerTime = 0;
const unsigned long pirCooldown = 10000;

// ==========================================
// BUTTON DEBOUNCING CONFIGURATION
// ==========================================
int lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// ==========================================
// PHOTO CAPTURE RATE LIMITING
// ==========================================
unsigned long lastPhotoTime = 0;
const unsigned long photoInterval = 200;

// ==========================================
// WEB SERVER SETUP
// ==========================================
WebServer server(80);

// ==========================================
// PIR STATE TRACKING
// ==========================================
int lastIrState = LOW;

// ==========================================
// BUZZER TRIGGER SOURCE TRACKING
// ==========================================
enum BuzzerSource { NONE, IR_SENSOR, BUTTON, WEB_SERVER };
BuzzerSource currentBuzzerSource = NONE;

// ==========================================
// DIAGNOSTIC MODE CONFIGURATION
// ==========================================
bool diagnosticMode = true;
unsigned long lastDiagnosticPrint = 0;
const unsigned long diagnosticInterval = 5000;  // Send diagnostics every 5 seconds

// ==========================================
// FUNCTION DECLARATIONS
// ==========================================
void handleBuzzer();
void sendPhoto();
void logBuzzerTrigger(String source);
String buildURL(const char* path);
void registerIPWithServer();
void printDiagnostics();
void sendLogToServer(String logMessage);  // NEW: Send log to server

// ==========================================
// SETUP - RUNS ONCE AT STARTUP
// ==========================================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(1000);
  Serial.println("\n\n");
  Serial.println("========================================");
  Serial.println("ESP32-CAM SECURITY SYSTEM");
  Serial.println("DIAGNOSTIC MODE ENABLED");
  Serial.println("========================================\n");

  // ------------------------------------------
  // BUZZER INITIALIZATION & TEST
  // ------------------------------------------
  Serial.println("--- BUZZER INITIALIZATION ---");
  sendLogToServer("SETUP: Starting buzzer initialization");
  
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW);
  Serial.println("✓ Buzzer pin 12 set as OUTPUT");
  sendLogToServer("SETUP: Buzzer pin 12 configured as OUTPUT");
  
  // Test buzzer
  Serial.println("Testing buzzer (3 short beeps)...");
  sendLogToServer("SETUP: Testing buzzer with 3 beeps");
  
  for (int i = 0; i < 3; i++) {
    digitalWrite(buzzerPin, HIGH);
    Serial.println("  Beep ON");
    delay(100);
    digitalWrite(buzzerPin, LOW);
    Serial.println("  Beep OFF");
    delay(100);
  }
  Serial.println("✓ Buzzer test complete\n");
  sendLogToServer("SETUP: Buzzer test completed successfully");

  // ------------------------------------------
  // BUTTON INITIALIZATION & TEST
  // ------------------------------------------
  Serial.println("--- BUTTON INITIALIZATION ---");
  sendLogToServer("SETUP: Starting button initialization on GPIO 14");
  
  pinMode(buttonPin, INPUT_PULLUP);
  Serial.println("✓ Button pin 14 set as INPUT_PULLUP");
  
  // Test button reading
  Serial.println("Testing button pin (5 readings)...");
  String buttonTestLog = "BUTTON_TEST: ";
  
  for (int i = 0; i < 5; i++) {
    int btnState = digitalRead(buttonPin);
    Serial.print("  Button reading ");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(btnState);
    Serial.println(btnState == HIGH ? " (HIGH - Not Pressed)" : " (LOW - PRESSED!)");
    
    buttonTestLog += String(btnState) + " ";
    delay(500);
  }
  
  Serial.println("✓ Button test complete\n");
  sendLogToServer(buttonTestLog);

  // ------------------------------------------
  // PIR SENSOR INITIALIZATION & TEST
  // ------------------------------------------
  Serial.println("--- PIR SENSOR INITIALIZATION ---");
  sendLogToServer("SETUP: Starting PIR sensor initialization on GPIO 13");
  
  pinMode(irSensorPin, INPUT);
  Serial.println("✓ PIR pin 13 set as INPUT");
  
  // Extended PIR diagnostic test
  Serial.println("Testing PIR sensor (20 readings over 10 seconds)...");
  int highCount = 0;
  int lowCount = 0;
  String pirTestLog = "PIR_TEST: ";
  
  for (int i = 0; i < 20; i++) {
    int pirState = digitalRead(irSensorPin);
    if (pirState == HIGH) highCount++;
    else lowCount++;
    
    Serial.print("  Reading ");
    Serial.print(i + 1);
    Serial.print("/20: ");
    Serial.print(pirState);
    Serial.println(pirState == HIGH ? " (HIGH)" : " (LOW)");
    
    pirTestLog += String(pirState);
    delay(500);
  }
  
  Serial.println("\n--- PIR TEST SUMMARY ---");
  Serial.print("HIGH readings: ");
  Serial.print(highCount);
  Serial.print(" (");
  Serial.print((highCount * 100) / 20);
  Serial.println("%)");
  Serial.print("LOW readings: ");
  Serial.print(lowCount);
  Serial.print(" (");
  Serial.print((lowCount * 100) / 20);
  Serial.println("%)");
  
  // Send PIR test summary to server
  String pirSummary = "PIR_SUMMARY: HIGH=" + String(highCount) + 
                      " (" + String((highCount * 100) / 20) + "%), LOW=" + 
                      String(lowCount) + " (" + String((lowCount * 100) / 20) + "%)";
  sendLogToServer(pirSummary);
  
  // Analyze PIR behavior
  if (highCount == 20) {
    Serial.println("\n⚠️ CRITICAL: PIR is STUCK HIGH!");
    sendLogToServer("PIR_ERROR: Sensor stuck HIGH - possible wiring issue");
  } else if (lowCount == 20) {
    Serial.println("\n✓ PIR is stable LOW (normal idle state)");
    sendLogToServer("PIR_STATUS: Stable LOW - normal idle state");
  } else {
    Serial.println("\n⚠️ WARNING: PIR is fluctuating");
    sendLogToServer("PIR_WARNING: Sensor fluctuating - may be warming up");
  }
  Serial.println("--- PIR TEST COMPLETE ---\n");

  // PIR warm-up period
  Serial.println("PIR Sensor warm-up (5 seconds)...");
  sendLogToServer("SETUP: PIR warm-up period starting");
  
  for (int i = 5; i > 0; i--) {
    Serial.print("  ");
    Serial.print(i);
    Serial.println("...");
    delay(1000);
  }
  Serial.println("✓ PIR warm-up complete\n");
  sendLogToServer("SETUP: PIR warm-up completed");

  // ------------------------------------------
  // WIFI CONNECTION
  // ------------------------------------------
  Serial.println("--- WIFI CONNECTION ---");
  sendLogToServer("SETUP: Starting WiFi connection");
  
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  int wifiAttempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifiAttempts < 40) {
    delay(500);
    Serial.print(".");
    wifiAttempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected");
    Serial.print("  IP Address: ");
    Serial.println(WiFi.localIP());
    
    String wifiLog = "WIFI: Connected successfully - IP: " + WiFi.localIP().toString();
    sendLogToServer(wifiLog);
    
    // Register IP with server
    Serial.println("\n--- IP REGISTRATION ---");
    registerIPWithServer();
  } else {
    Serial.println("\n⚠️ WiFi connection FAILED!");
    sendLogToServer("WIFI_ERROR: Connection failed after " + String(wifiAttempts) + " attempts");
  }

  // ------------------------------------------
  // WEB SERVER INITIALIZATION
  // ------------------------------------------
  server.on("/buzzer", HTTP_POST, handleBuzzer);
  server.begin();
  Serial.println("\n✓ HTTP server started on port 80");
  sendLogToServer("SETUP: HTTP server started on port 80");

  // ------------------------------------------
  // CAMERA CONFIGURATION
  // ------------------------------------------
  Serial.println("\n--- CAMERA INITIALIZATION ---");
  sendLogToServer("SETUP: Starting camera initialization");
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("✗ Camera init failed! Error 0x%x\n", err);
    sendLogToServer("CAMERA_ERROR: Initialization failed - Error code: 0x" + String(err, HEX));
    while(1) {
      delay(1000);
    }
  }
  Serial.println("✓ Camera initialized successfully!");
  sendLogToServer("SETUP: Camera initialized successfully");

  // Read initial states
  lastIrState = digitalRead(irSensorPin);
  lastButtonState = digitalRead(buttonPin);
  
  String initialStates = "INITIAL_STATE: Button=" + String(lastButtonState) + 
                         " PIR=" + String(lastIrState);
  sendLogToServer(initialStates);
  
  Serial.println("\n========================================");
  Serial.println("SETUP COMPLETE - SYSTEM READY");
  Serial.println("========================================");
  sendLogToServer("SETUP: System initialization completed - Ready for operation");
  
  Serial.println("\nMonitoring sensors...");
  Serial.println("- Diagnostics sent to server every 5 seconds");
  Serial.println("- Press button or trigger PIR to test\n");
}

// ==========================================
// MAIN LOOP
// ==========================================
void loop() {
  server.handleClient();

  // Send diagnostics to server periodically
  if (diagnosticMode && (millis() - lastDiagnosticPrint >= diagnosticInterval)) {
    printDiagnostics();
    lastDiagnosticPrint = millis();
  }

  // ==========================================
  // BUTTON HANDLING
  // ==========================================
  int buttonReading = digitalRead(buttonPin);
  
  if (buttonReading != lastButtonState) {
    lastDebounceTime = millis();
    
    String buttonLog = "BUTTON_STATE_CHANGE: " + 
                       String(lastButtonState) + " → " + String(buttonReading);
    Serial.println("[BUTTON] State changed: " + String(lastButtonState) + " → " + String(buttonReading));
    sendLogToServer(buttonLog);
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (buttonReading == LOW && lastButtonState == HIGH && !buzzerOn) {
      Serial.println("\n╔════════════════════════════════╗");
      Serial.println("║   BUTTON PRESS DETECTED!       ║");
      Serial.println("╚════════════════════════════════╝");
      
      sendLogToServer("BUTTON_TRIGGER: Button pressed - activating buzzer for 500ms");
      
      digitalWrite(buzzerPin, HIGH);
      buzzerOn = true;
      buzzerOnTime = millis();
      currentBuzzerSource = BUTTON;
      
      Serial.println("→ Buzzer activated for 500ms");
      logBuzzerTrigger("Button");
    }
  }
  lastButtonState = buttonReading;

  // ==========================================
  // PIR SENSOR HANDLING
  // ==========================================
  int currentIrState = digitalRead(irSensorPin);

  if (currentIrState != lastIrState) {
    String pirLog = "PIR_STATE_CHANGE: " + 
                    String(lastIrState) + " → " + String(currentIrState);
    Serial.print("[PIR] State changed: ");
    Serial.println(pirLog);
    sendLogToServer(pirLog);
  }

  if (currentIrState == HIGH && lastIrState == LOW && !buzzerOn) {
    unsigned long timeSinceLastTrigger = millis() - lastPirTriggerTime;
    
    if (timeSinceLastTrigger >= pirCooldown) {
      Serial.println("\n╔════════════════════════════════╗");
      Serial.println("║   PIR MOTION DETECTED!         ║");
      Serial.println("╚════════════════════════════════╝");
      
      String pirTriggerLog = "PIR_TRIGGER: Motion detected - activating buzzer for 500ms (Time since last: " + 
                             String(timeSinceLastTrigger / 1000) + "s)";
      sendLogToServer(pirTriggerLog);
      
      digitalWrite(buzzerPin, HIGH);
      buzzerOn = true;
      buzzerOnTime = millis();
      currentBuzzerSource = IR_SENSOR;
      lastPirTriggerTime = millis();
      
      delay(1500); 
      
      Serial.println("→ Buzzer activated for 500ms");
      Serial.println("→ Photo capture enabled");
      logBuzzerTrigger("IR Sensor");
    } else {
      unsigned long remainingTime = (pirCooldown - timeSinceLastTrigger) / 1000;
      String cooldownLog = "PIR_COOLDOWN: Motion ignored - " + 
                          String(remainingTime) + "s remaining";
      Serial.println("[PIR] " + cooldownLog);
      sendLogToServer(cooldownLog);
    }
  }
  
  lastIrState = currentIrState;

  // ==========================================
  // BUZZER AUTO-OFF HANDLING
  // ==========================================
  if (buzzerOn) {
    if (currentBuzzerSource == IR_SENSOR) {
      if (millis() - lastPhotoTime >= photoInterval) {
        sendPhoto();
        lastPhotoTime = millis();
      }
    }

    if (millis() - buzzerOnTime >= buzzerDuration) {
      digitalWrite(buzzerPin, LOW);
      buzzerOn = false;
      
      String source = "";
      if (currentBuzzerSource == BUTTON) source = "Button";
      else if (currentBuzzerSource == IR_SENSOR) source = "PIR";
      else if (currentBuzzerSource == WEB_SERVER) source = "WebAPI";
      
      String buzzerOffLog = "BUZZER_OFF: Auto-off after 500ms (Source: " + source + ")";
      Serial.println("\n[BUZZER] " + buzzerOffLog);
      sendLogToServer(buzzerOffLog);
      
      currentBuzzerSource = NONE;
    }
  }
}

// ==========================================
// NEW: SEND LOG TO SERVER
// ==========================================
/*
 * Sends diagnostic log messages to Python FastAPI server
 * Endpoint: POST /esp32-log
 * Payload: {"log": "message"}
 */
void sendLogToServer(String logMessage) {
  // Only send if WiFi is connected
  if (WiFi.status() != WL_CONNECTED) {
    return;  // Silently fail if no WiFi
  }

  HTTPClient http;
  String url = buildURL(diagnosticLogPath);
  
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);  // 3 second timeout for logs
  
  // Build JSON payload: {"log": "message"}
  // Escape quotes in log message
  logMessage.replace("\"", "\\\"");
  String payload = "{\"log\":\"" + logMessage + "\"}";
  
  int httpResponseCode = http.POST(payload);
  
  // Only print errors to avoid log spam
  if (httpResponseCode < 0) {
    Serial.printf("[LOG] Failed to send: %s\n", http.errorToString(httpResponseCode).c_str());
  }
  
  http.end();
}

// ==========================================
// PRINT DIAGNOSTICS (SEND TO SERVER)
// ==========================================
void printDiagnostics() {
  Serial.println("\n--- SENSOR STATUS ---");
  
  // Build diagnostic message
  int btnState = digitalRead(buttonPin);
  int pirState = digitalRead(irSensorPin);
  
  String diagnosticMsg = "DIAGNOSTICS: Button=" + String(btnState) + 
                         " (" + String(btnState == HIGH ? "NotPressed" : "PRESSED") + 
                         "), PIR=" + String(pirState) + 
                         " (" + String(pirState == HIGH ? "Motion" : "NoMotion") + 
                         "), Buzzer=" + String(buzzerOn ? "ON" : "OFF");
  
  // Add PIR cooldown info
  if (lastPirTriggerTime > 0) {
    unsigned long timeSince = millis() - lastPirTriggerTime;
    if (timeSince < pirCooldown) {
      unsigned long remaining = (pirCooldown - timeSince) / 1000;
      diagnosticMsg += ", PIR_Cooldown=" + String(remaining) + "s";
    } else {
      diagnosticMsg += ", PIR_Cooldown=Ready";
    }
  }
  
  Serial.print("Button (GPIO 14): ");
  Serial.println(btnState == HIGH ? "HIGH (Not Pressed)" : "LOW (PRESSED!)");
  
  Serial.print("PIR (GPIO 13):    ");
  Serial.println(pirState == HIGH ? "HIGH (Motion!)" : "LOW (No Motion)");
  
  Serial.print("Buzzer (GPIO 12): ");
  Serial.println(buzzerOn ? "ON" : "OFF");
  
  if (lastPirTriggerTime > 0) {
    unsigned long timeSince = millis() - lastPirTriggerTime;
    if (timeSince < pirCooldown) {
      unsigned long remaining = (pirCooldown - timeSince) / 1000;
      Serial.print("PIR Cooldown:     ");
      Serial.print(remaining);
      Serial.println("s remaining");
    } else {
      Serial.println("PIR Cooldown:     Ready");
    }
  }
  
  Serial.println("--------------------\n");
  
  // Send to server
  sendLogToServer(diagnosticMsg);
}

// ==========================================
// HELPER FUNCTIONS
// ==========================================
String buildURL(const char* path) {
  return String("http://") + serverIP + ":" + serverPort + path;
}

void registerIPWithServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("  ⚠ Cannot register IP - WiFi not connected");
    sendLogToServer("IP_REGISTRATION: Failed - WiFi not connected");
    return;
  }

  IPAddress localIP = WiFi.localIP();
  String ipString = localIP.toString();
  
  Serial.print("  Registering IP: ");
  Serial.println(ipString);

  HTTPClient http;
  String url = buildURL(ipRegistrationPath);
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  
  String payload = "{\"ip\":\"" + ipString + "\"}";
  int httpResponseCode = http.POST(payload);
  
  if (httpResponseCode > 0) {
    Serial.printf("  ✓ Registered (HTTP %d)\n", httpResponseCode);
    String response = http.getString();
    Serial.print("  Response: ");
    Serial.println(response);
    sendLogToServer("IP_REGISTRATION: Success - IP " + ipString + " registered");
  } else {
    Serial.printf("  ✗ Failed: %s\n", http.errorToString(httpResponseCode).c_str());
    sendLogToServer("IP_REGISTRATION: Failed - " + String(http.errorToString(httpResponseCode).c_str()));
  }
  
  http.end();
}

void logBuzzerTrigger(String source) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = buildURL(logPath);
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    
    String payload = "{\"event\":\"" + source + " triggered\", \"timestamp\":" + String(millis()) + "}";
    int httpResponseCode = http.POST(payload);
    
    if (httpResponseCode > 0) {
      Serial.printf("  ✓ Log sent (HTTP %d)\n", httpResponseCode);
    } else {
      Serial.printf("  ✗ Log failed: %s\n", http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  }
}

void handleBuzzer() {
  Serial.println("\n[WEB API] Buzzer request received");
  sendLogToServer("WEBAPI: Buzzer endpoint called");
  
  if (server.hasArg("state")) {
    String state = server.arg("state");
    Serial.print("  State: ");
    Serial.println(state);
    
    if (state == "on") {
      digitalWrite(buzzerPin, HIGH);
      buzzerOn = true;
      buzzerOnTime = millis();
      currentBuzzerSource = WEB_SERVER;
      Serial.println("  ✓ Buzzer ON");
      sendLogToServer("WEBAPI: Buzzer turned ON via API");
      server.send(200, "text/plain", "Buzzer ON (auto-off in 500ms)");
    } else if (state == "off") {
      digitalWrite(buzzerPin, LOW);
      buzzerOn = false;
      currentBuzzerSource = NONE;
      Serial.println("  ✓ Buzzer OFF");
      sendLogToServer("WEBAPI: Buzzer turned OFF via API");
      server.send(200, "text/plain", "Buzzer OFF");
    } else {
      Serial.println("  ✗ Invalid state");
      sendLogToServer("WEBAPI: Invalid state parameter - " + state);
      server.send(400, "text/plain", "Invalid state");
    }
  } else {
    Serial.println("  ✗ Missing state parameter");
    sendLogToServer("WEBAPI: Missing state parameter");
    server.send(400, "text/plain", "Missing state parameter");
  }
}

void sendPhoto() {
  Serial.println("  📷 Capturing...");
  
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("  ✗ Capture failed");
    sendLogToServer("CAMERA: Photo capture failed");
    return;
  }

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = buildURL(uploadPath);
    http.begin(url);
    http.setTimeout(10000);
    http.addHeader("Content-Type", "image/jpeg");
    
    int httpResponseCode = http.POST(fb->buf, fb->len);
    
    if (httpResponseCode > 0) {
      Serial.printf("  ✓ Uploaded (HTTP %d, %d bytes)\n", httpResponseCode, fb->len);
      sendLogToServer("CAMERA: Photo uploaded successfully - " + String(fb->len) + " bytes");
    } else {
      Serial.printf("  ✗ Failed: %s\n", http.errorToString(httpResponseCode).c_str());
      sendLogToServer("CAMERA: Photo upload failed - " + String(http.errorToString(httpResponseCode).c_str()));
    }
    http.end();
  }

  esp_camera_fb_return(fb);
}
