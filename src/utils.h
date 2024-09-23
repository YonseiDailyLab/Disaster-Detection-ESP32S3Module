#ifndef UTILS_H
#define UTILS_H

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#define SUCSESSED {0, 255, 0}
#define INIT {127, 0, 127}
#define CONNECTING {0, 0, 255}
#define ERROR {255, 0, 0}
#define WARNING {127, 127, 0}
#define RECV {0, 127, 127}
#define TRANSMIT {127, 127, 127}
#define TURN_OFF {0, 0, 0}

typedef struct{
    uint8_t r;
    uint8_t g;
    uint8_t b;
}Color;

void set_pixels_mode(Adafruit_NeoPixel &pixels, Color color);
void set_pixels_blank(Adafruit_NeoPixel &pixels, Color color,  int duration, int rotation);

#endif // UTILS_H