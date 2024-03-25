# Owleye
## Intro
Owleye gives you the posibiliy to transform your webcam to an eye tracker. At first, you should calibrate owleye to know you, then it detects which point you are looking on your monitor. 
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

This is the opened window:

![Screenshot 2024-03-13 013738](https://github.com/MustafaLotfi/Owleye/assets/53625380/9e0996ed-560b-4329-b101-1496e51ffb17)

You can learn about the program's usage in [this tutorial](https://github.com/MustafaLotfi/Owleye/blob/main/docs/USE_APP.md).

## Method

**Owleye's structure:**

![Screenshot 2024-03-25 024008](https://github.com/MustafaLotfi/Owleye/assets/53625380/c24f43f5-ed99-48e0-9a96-69b20536f240)

While the camera is streaming, Owleye gets the images and extracts head and eyes features. Then it feeds this data to the neural networks (NN) models to calculate the user's eye viewpoint.

### Input

As in the first block of the [Owleye's structure](https://private-user-images.githubusercontent.com/53625380/316467756-c24f43f5-ed99-48e0-9a96-69b20536f240.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTEzNjM3MzksIm5iZiI6MTcxMTM2MzQzOSwicGF0aCI6Ii81MzYyNTM4MC8zMTY0Njc3NTYtYzI0ZjQzZjUtZWQ5OS00OGUwLTlhOTYtNjliMjA1MzZmMjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzI1VDEwNDM1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMzYzZlNDg5YzU3N2YyMDZkYTQyYjJiODdhMWQ2OTU2Y2ZiZDlkZWRiZmM5Mjk3ODg5YTBhY2NmYzIzMDRkNjkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.yaBCEz1DDeiMPQw_XU8gR4NbjKtGShJqZ7XiQ8AKrkk) is visible, it receives the user's images during time and after detecting thier face, in the second block it extracts their 478 landmarks/keypoints. It's done by canonical face model which is in the world coordinates. Owleye uses Mediapipe package to implement these steps. Then in the third block, Owleye compute the face rotation and position vectors by extracted landmarks. In the fourth block, Owleye extracts the eyes' images using landmarks and gives them to the fifth block to calculate iris positions. Finlaly, three type of inputs are ready to be fed to sixth block which is eye viewpoint predictive model:
- **Head rotation and position vectors:** (r1, r2, r3), (x, y, z). Rotation and position, world coordinates.
- **Left and right eyes iris:** (xl, yl), (xr, yr). These are calculated respect to the eyes (image coordinates).
- **Eyes images:** Two images are concatenated together in rows.

We will consider the first and the second inputs as the **face vector** which has a length of 10.

![Screenshot 2024-03-14 034920](https://github.com/MustafaLotfi/Owleye/assets/53625380/b1f44929-a867-45eb-b5be-211c5f41f08c)

### Output

The output of Owleye is a vector of user's eye view points on screen (xp, yp) per sample (an image and a vector). During time, this output will be a matrix. The matrix's shape is n by 2. The values are normalized between 0 and 1. For example, the program tracks the user for 10 seconds, with an FPS of 15, we have a matirx with a shape of 150 by 2. The first column is for the horizontal direction and the second is for the vertical direction.

### Calibration
The calibration process consists of looking at a white point in a black screen for a certain time. Then, the point's position changes and the user must look at it again. This process is repeated until the calibration ends. During this procedure, Owleye collects data (input and output). It means each sample data entails one image, one face vector and one location point. This is because we already have the first five blocks, and the models and calculations have been prepared. Just the sixth block should be made.

### Dataset

We implemented calibration on 20 male subjects and collected 221000 samples (eyes images and face vectors as inputs and appeared point locations as outputs).

### Modeling

For the sixth block in [Owleye's structure](https://private-user-images.githubusercontent.com/53625380/316467756-c24f43f5-ed99-48e0-9a96-69b20536f240.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTEzNjM3MzksIm5iZiI6MTcxMTM2MzQzOSwicGF0aCI6Ii81MzYyNTM4MC8zMTY0Njc3NTYtYzI0ZjQzZjUtZWQ5OS00OGUwLTlhOTYtNjliMjA1MzZmMjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzI1VDEwNDM1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMzYzZlNDg5YzU3N2YyMDZkYTQyYjJiODdhMWQ2OTU2Y2ZiZDlkZWRiZmM5Mjk3ODg5YTBhY2NmYzIzMDRkNjkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.yaBCEz1DDeiMPQw_XU8gR4NbjKtGShJqZ7XiQ8AKrkk), Two Convolutional Neural Network (CNN) models are used to predict the user's eye view point in the horizonal and vertical directions on the monitor. These models are trained using the dataset. We called them "base models".

**Network architecture:**
![Screenshot 2024-03-16 163427](https://github.com/MustafaLotfi/Owleye/assets/53625380/02d196c2-c9c2-497d-b1e5-d3d7b2a29160)

Right side of the above picture illustrates the CNN model's structure. In this model there is two branches. The left branch is for the eyes image, and the right branch is for a vector with a length of 10. six value for head's rotation and position and 4 value for iris position.

### Fine-tuning

To customize Owleye for each person, we considered the approach of retraining the models of [the sixth block](https://private-user-images.githubusercontent.com/53625380/316467756-c24f43f5-ed99-48e0-9a96-69b20536f240.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTEzNjM3MzksIm5iZiI6MTcxMTM2MzQzOSwicGF0aCI6Ii81MzYyNTM4MC8zMTY0Njc3NTYtYzI0ZjQzZjUtZWQ5OS00OGUwLTlhOTYtNjliMjA1MzZmMjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzI1VDEwNDM1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMzYzZlNDg5YzU3N2YyMDZkYTQyYjJiODdhMWQ2OTU2Y2ZiZDlkZWRiZmM5Mjk3ODg5YTBhY2NmYzIzMDRkNjkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.yaBCEz1DDeiMPQw_XU8gR4NbjKtGShJqZ7XiQ8AKrkk) for each person. During this, data is collected from the person who we want to get their eye movements. the amount of data collected is not as much as the main dataset that we explained above. So, the last layer's weights of the base models in the sixth block change based on the new collected data. In this way, the network maintains its original shape and just is calibrated a little for each person.

### Fixations

The IV-T method is used to extract user's fixations. A [fixation](https://en.wikipedia.org/wiki/Fixation_(visual)) is a series of eye view points that are close together. So, first of all we removed the outliers using median filter. Then we merged close fixations and removed short ones. below image shows the fixations in a monitor during a certain time.

![Screenshot 2024-03-16 195233](https://github.com/MustafaLotfi/Owleye/assets/53625380/57b9e984-5f54-48a0-984e-110c65b2cc20)


### Blinking

Indeed, while the user is blinking, they aren't seeing anywhere. So, the data in that short time should be removed. We've calculated the blinking using Eye Aspect Ratio (EAR) method. In this way, when the user's EAR goes lower than a certain threshold, it is considered as a blink. So, the output (x, y) will be deleted in the next computations. Also in this periods, we can interpolate the outputs during time.

![Screenshot 2024-03-16 201348](https://github.com/MustafaLotfi/Owleye/assets/53625380/9ba0751f-ac96-46cd-a878-053a7e55158c)


## Limitations and future works
**1) Recunstructing whole code:** The structure of the code is terrible:)) Owleye is made in 2021 and I have not dedicated so much of time to improve it since then. Therefore, a lot of things have changed. The structure of the code totally can be redesigned to reach a better performance. The code can be more object oriented. the libraries (mediapipe and tensorflow) have changed a lot. So, the algorithm can be rewritten considering the changes.

**2) Changing the calibration algorithm:** The calibration duration time is really long. Using methods like image morphing makes it unnecessary to collect images in all positions and angles of the head and eyes.

**3) Changing the fine-tuning method:** In the current method, to retrain the algorithm, we considered to just change the weights in the last layer of the network. This fine-tuning process can be improved by implementing better solutions.

**4) Adding camera calibration:** The computed head angles and positions are meaningful since the camera is no calibrated. By calibrating and having angles and positions of the head, we can calculate the real eyes' angles and positions. So, using these parameters, implement better methods for reaching to the eyes view point. Maybe just with a simple linear regression model and real parameters of the head and eyes we could reach to the target.

**5) Creating a python library:** It can be desired to create a package from the code. So, everybody could just install and import the library and use it as they want.

**6) Providing real-time usage:** For now, it isn't possible to use the program in real-time. Because the FPS goes down in this way. the program's FPS for a camera that is 30 FPS reaches to 15. So, by optimizing some packages, we can get to a better result.

## Contributing

Feel free to improve the project. I'll appreciate your pull requests.

## About project

Owleye is a subsection of my master thesis: "Driver's Hazard Perception Assessment in a Driving Simulator".
