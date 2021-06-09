# IP
Image Processing

Using:
- Python 3.7
- openCV, sci-kit image

Detecting LEDs in a image of a breadboard with mounted LEDs.

Steps:
1. resizing
2. grayscale
3. applying blur
4. thresholding (bright spots -> white; rest -> black)
5. searching and counting light sources
6. marking those spots in a normal image
