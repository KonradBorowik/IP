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

- [x] Detecting 3 LEDs and deciding the orientation of an object
* Detects 3 LEDs in a picture
* connects them into a triangle isosceles triangle
* using pca alorythm I can get the middle point of the trainlge and it's orientation in regard to hypotenuse

- [ ] Following an object in a video


<i>the code is a mess but right now I'm focusing on achievieng my goal</i>
