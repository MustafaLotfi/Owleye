# Owleye
## Intro
Owleye gives you the posibiliy to transform your webcam to an eye tracker. Owleye is a subsection of my master thesis: "Driver's Hazard Perception Assessment in a Driving Simulator".y
At first, you should calibrate your camera, then the program tells you which point you are looking on your monitor. Indeed, this is a top-table eye tracker.
___
## Installation

### 1.Use source code:

Open terminal, download the repo:     `git clone <repo address>`

(or just download the zip file)

Go to the project directory:     `cd Owleye`

make a virtual environment:     `python -m venv env` or `virtualenv env`

activate the virtual environment:

Windows: `./env/Scripts/activate`

Linux: `source env/bin/activate`

Install required libraries:    `pip install -r requirements.txt`

Run the program:     `python main.py`

### 2. Use .exe file

Download the release file. It is tested on Windows 10.

Run Owleye.exe

## Usage

After activating virtual environment, run main.py

`python main.py`

or if you have downloaded the Owleye.exe, run it.

In the opened window, there are some parameters that you can change:

![Screenshot 2024-03-12 191947](https://github.com/MustafaLotfi/Owleye/assets/53625380/f7f14723-0dd4-4fea-b4ae-ff51b0a59654)


## Method
While the camera is streaming, Owleye gets the images and extracts the user's face rotation (r1, r2, r3) and position (x, y, z) vectors using mediapipe and opencv. Also it extracts the driver's iris positions (x1, y1), (x2, y2). Alongside with this, Owleye extract the eyes images and add them together. So, an input of one image (two eyes) and one vector (10 scalar) is ready to calculate the location that the user is looking. To do this, two Convolutional Neural Network models (CNNs) are used to predict the user's eye view point in the horizonal and vertical directions on the monitor. These models are trained on 221000 samples (images and vectors).

### Dataset
221000 samples (eye images and vectors) are collected from 20 subjects. The subjects were told look at the red point.

### Modeling

### Calibration
The calibration process consists of looking at a circle in the screen at a certain time.


## Limitations and future works
**1) Recunstructing whole code:** The structure of the code is terrible. Owleye is made in 2021. Therefore, a lot of things have changed since then. The structure of the code totally can be redesigned to reach a better performance. The code can be more object oriented. the libraries (mediapipe and tensorflow) have changed a lot. So, the algorithm can be rewritten considering the changes.

**2) Changing the calibration algorithm:** The calibration duration time is really long. Using methods like image morphing makes it unnecessary to collect images from all positions and angles.

**3) Changing the fine-tuning method:** In the current method, to retrain the algorithm, we considered to just change the weights in the last layer of the network. In this way, the network keeps the original shape of itself and just changes the last layer's weights to customize the network for each subject. But, this fine-tuning process can be improved by implementing better solutions.
