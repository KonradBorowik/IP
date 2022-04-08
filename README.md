# IP
Image Processing

## Using:
* openCV
* sci-kit image
* imutils

- [x] Detecting LEDs in a image of a breadboard with mounted LEDs.
* ### Steps:
  1. resizing
  2. grayscale 
  3. applying blur
  4. thresholding (bright spots -> white; rest -> black)
  5. searching and counting light sources
  6. marking those spots in a normal image

- [ ] Detecting color of the LED
* I've got an idea for that, but will try later
* UPDATE: it wasn't working good enough

- [x] Detecting 3 LEDs and deciding the orientation of the object
* Detects 3 LEDs in a picture
* ~~connects them into a isosceles triangle~~
* ~~using pca alorythm I can get the middle point of the trainlge and it's orientation in regard to hypotenuse~~
* asociates them with triangle apexes and sides
* then decides the shortest base
* calculates angle between global x axis and the triangle's height (the one that falls on the shortest base)

- [x] Following an object in live camera feed
* I assume every frame as a separate image
* then, every image is being processed by the above function (the one that detects 3 LEDs and orientation in a picture)
* program instructs user how to move the object to complete a route

Video: https://youtu.be/k-LYcO7BcV8
