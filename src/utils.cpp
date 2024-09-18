#include "utils.h"

void set_pixels_mode(Adafruit_NeoPixel &pixels, Color color){
    pixels.setPixelColor(0, pixels.Color(color.r, color.g, color.b));
    pixels.show();
}

void set_pixels_blank(Adafruit_NeoPixel &pixels, Color color,  int duration, int rotation){
  for(int i = 0; i < rotation; i++){
    set_pixels_mode(pixels, color);
    delay(duration);
    set_pixels_mode(pixels, TURN_OFF);
    delay(duration);
  }
}