# Owleye
## Intro
Owleye gives you the posibiliy to transform your webcam to an eye tracker. At first, you should calibrate owleye to know you, then it tells you which point you are looking on your monitor. 
___
## Installation

### 1.Use source code:

Open terminal, download the repo:
`git clone <repo address>`

(or just download the zip file)

Go to the project directory:
`cd Owleye`

make a virtual environment:
`python -m venv env` or `virtualenv env`

activate the virtual environment:

Windows:
`.\env\Scripts\activate`

Linux:
`source env/bin/activate`

Install required libraries:
`pip install -r requirements.txt`

### 2. Use .exe file

Download the release file. It is tested on Windows 10.

## Usage

If you are using the source code, after activating the virtual environment, run main.py:
`python main.py`

or if you have downloaded the Owleye.exe, run it.

In the opened window, there are some parameters that you can change:

![Screenshot 2024-03-13 013738](https://github.com/MustafaLotfi/Owleye/assets/53625380/9e0996ed-560b-4329-b101-1496e51ffb17)

## Method

While the camera is streaming, Owleye gets the images and extracts head and eyes features. Then it feeds this data to the neural networks models to calculate the user's eye view point.

### Input

Owleye receives the user's images during time and extracts their face 478 landmarks/keypoints using Mediapipe library. It's done by canonical face model which is in the world coordinates. Then Owleye extracts below data using the landmarks:
- **Head rotation and position vectors:** (r1, r2, r3), (x, y, z) are calculated using Opencv library
- **Left and right eyes iris:** (xl, yl), (xr, yr). These are calculated respect to the eyes
- **Eyes images:** Two images are concatenated together in rows.

![Screenshot 2024-03-14 034920](https://github.com/MustafaLotfi/Owleye/assets/53625380/b1f44929-a867-45eb-b5be-211c5f41f08c)


Now, an input of one image (two eyes) and one vector (10 scalar) is ready to calculate the target.

### Output

The output of Owleye is matrix of user's eye view points on screen (xp, yp) during time. The matrix's shape is n by 2. The values are normalized between 0 and 1. For example, the program tracks the user for 10 seconds, with an FPS of 15, we have a matirx with a shape of 150 by 2. The first column is for the horizontal direction and the second is for the vertical direction.

### Calibration
The calibration process consists of looking at a white point in a black screen for a certain time. Then, the point's position changes and the user must look at it again. This process is repeated until the calibration ends. During this procedure, Owleye collects data (input and output). It means each sample data entails one image, one vector and one location point.

### Dataset

We implemented calibration on 20 male subjects and collected 221000 samples (eye images and vectors).

### Modeling

Two Convolutional Neural Network (CNNs) models are used to predict the user's eye view point in the horizonal and vertical directions on the monitor. These models are trained using the dataset. We called them "base models".

**Network architecture:**
![Screenshot 2024-03-16 163427](https://github.com/MustafaLotfi/Owleye/assets/53625380/02d196c2-c9c2-497d-b1e5-d3d7b2a29160)

### Fine-tuning

To customize two base models for each person, we considered a retraining process. During this, data is collected from the person who we want to track their point of view. the amount of data collected is not as much as the main dataset. So, the last layer's weights change based on the new collected data. In this way, the network retains its original shape and just is calibrated a little for each person.

### Fixations

The IV-T method is used to extract user's fixations.

### Blinking

Indeed, while the user is blinking, they aren't seeing anywhere. So, the data in that short time should be removed. We've calculated the blinking using Eye Aspect Ratio (EAR) method. In this way, when the user's EAR goes lower than a certain threshold, it is considered as a blink. So, the output (x, y) will be deleted in the next computations. Also in this periods, we can interpolate the outputs during time.

![Screenshot 2024-03-16 193621](https://github.com/MustafaLotfi/Owleye/assets/53625380/0313304e-902d-45b5-b977-f81954b7f91d)


## Limitations and future works
**1) Recunstructing whole code:** The structure of the code is terrible:)) Owleye is made in 2021. Therefore, a lot of things have changed since then. The structure of the code totally can be redesigned to reach a better performance. The code can be more object oriented. the libraries (mediapipe and tensorflow) have changed a lot. So, the algorithm can be rewritten considering the changes.

**2) Changing the calibration algorithm:** The calibration duration time is really long. Using methods like image morphing makes it unnecessary to collect images in all positions and angles of the head and eyes.

**3) Changing the fine-tuning method:** In the current method, to retrain the algorithm, we considered to just change the weights in the last layer of the network. This fine-tuning process can be improved by implementing better solutions.

**4) Adding camera calibration:** The computed head angles and positions are meaningful since the camera is no calibrated. By calibrating and having angles and positions of the head, we can calculate the real eyes' angles and positions. So, using these parameters, implement better methods for reaching to the eyes view point. Maybe just with a simple linear regression model and real parameters of the head and eyes we could reach to the target.

**5) Creating a python library:** It can be desired to create a package from the code. So, everybody could just install and import the library and use it as they want.

**6) Providing real-time usage:** For now, it isn't possible to use the program in real-time. Because the FPS goes down in this way. the program's FPS for a camera that is 30 FPS reaches to 15. So, by optimizing some packages, we can get to a better result.

## Contributing

Feel free to improve the project. I'll appreciate your pull requests.

## About project

Owleye is a subsection of my master thesis: "Driver's Hazard Perception Assessment in a Driving Simulator".
