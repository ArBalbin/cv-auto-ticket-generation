// arduino/buzzer_alert.ino
int buzzer = 8;
int led = 13;

void setup() {
  pinMode(buzzer, OUTPUT);
  pinMode(led, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '0') { // low
      digitalWrite(buzzer, LOW);
      digitalWrite(led, LOW);
    } else if (c == '1') { // medium
      digitalWrite(buzzer, LOW);
      digitalWrite(led, HIGH);
    } else if (c == '2') { // high
      // beep buzzer
      for (int i=0; i<5; i++) {
        digitalWrite(buzzer, HIGH);
        delay(200);
        digitalWrite(buzzer, LOW);
        delay(200);
      }
      digitalWrite(led, HIGH);
    }
  }
}
