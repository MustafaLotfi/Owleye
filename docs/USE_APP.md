## This document shows you how to use Owleye application

When you run the program, a window like the one in below will be appeared:

![Screenshot 2024-03-25 142116](https://github.com/MustafaLotfi/Owleye/assets/53625380/1d311246-4273-4092-a1ab-d7dd2b84173f)

Also, after running, a folder called "subjects" will be created. In this folder, a folder will be created for each subject based on the subject number in the UI. Then, you can adjust the items that you need, and finally "start" the program. In following, the items will be explained.

**1. Subject number:** A specific number that you enter as the ID of the subject.

**2. Camera ID:** Usually it is 0, but you can try other numbers if you have several webcams on your system.

**3. Camera:** by activating this checkbox, after clicking on the "start" button, the webcam stream will be shown. So you can see yourself and the landmarks that already are detected on your face.

**4. Calibration:** By activating this checkbox, after clicking on the "start" button, the program will be ready to collect data (inputs and outputs of the models of [the sixth block in owleye's structure](https://github.com/MustafaLotfi/Owleye/blob/main/docs/images/Owleye%20structure.png)) from the user. So, a white point will be appeared on the screen. As soon as you press the "SPACE" key on keyboard, the program starts collecting data for a particlular time. While this, the background bacomes black and the user should look at the white point during this time. If the data collection ends, the screen will become gray and the point will go to another location. The user can look everywhere and actually rest in this situation. Again, the "SPACE" key should be pressed for data collection and all of the explained process get repeated. Also, a folder called "clb" will be created in the user's folder. In this folder, 5 ".pickle" files will be created which were collected:

- t.pickle: Time
- x1.pickle: Eyes images
- x2.pickle: Face vecotrs
- y.pickle: White point locations
- er.pickle: Eye aspect ratio vector (Go to [11th section](https://github.com/MustafaLotfi/Owleye/blob/main/docs/USE_APP.md#:~:text=from%20the%20default.-,11.%20Threshold,-%3A%20To%20detect) to know about this)

**5. Subject name:** This item is arbitrary. you can enter the user's name.

**6. Describtion:** This section is arbitrary too. you can enter any information that your user has.

**7. Calibration grid:** This item can have three types of integer numbers.

- 2 numbers (n, c): The white point starts to move just horizontally in the screen in n rows. Each row contains c locations that the point goes through.
- 3 numbers (n, m , c): The white point does not move. just in a grid by size of n x m (like a matrix) remains fixed in each location.
- 4 numbers (n, c, m, d): The white point moves both horizontally and vertically, in n rows with c locations and m columns with d locations.

**8. Sampling:** If you activate this checkbox, the program will start collecting data from you while you are looking in the screen. So, this item is for using the program for your goal. Also a folder called smp will be created in the user's folder. In this folder these files will be made:
- t.pickle
- sys_time.pickle
- x1.pickle
- x2.pickle
- er.pickle

**9. Testing:** This checkbox is for seeing how well Owleye works. If you activate this checkbox, after starting the program, It will start showing you a white point that you must look at that. Actually, the user is looking in the white point. So, it is clear that what should be the best possible prediction of Owleye (The position of white points). Also, you can see mean squared error (MSE).

**10. Tune blinking threshold:** By activating this checkbox, you can change the blinking threshold (11) from the default.

**11. Threshold:** To detect the blinks, The eyes aspect ratio (EAR) method is used. In the samples that are collected during sampling, EAR will be calculated. It is a vector during time. Then using that vector, the velocity of EAR will be calculated as a vector. So, the values above the determined threshold are considered as a blink. The default value for threshold is gained using try and error with my face. every face can have a different threshold.

**12. Tune eye tracking model:** By activating this checkbox, after pressing "start" button, the program starts retraining the two horizontal and vertical models of [the sixth block in the Owleye's structure](https://github.com/MustafaLotfi/Owleye/blob/main/docs/images/Owleye%20structure.png). After this, a folder called "mdl" will be created in the subject's folder. In this folder, there are 3 files:
- mdl1.pickle: scaler of the face vector
- mdl1-hrz.pickle: model for predicting the horizontal direction
- mdl1-vrt.pickle: model for predicting the vertical direction

**13. SS:** Abbreviation of shift samples. While the white point is moving during calibration, the inputs and outputs (point locations) are not exactly aligned. Because of processing problems, the images (inputs) are a little later than the outputs. For example, if this parameter is equal to 20, it means you want to shift inputs in 20 samples. So, the 21th input will be aligned with 1st output.

**14. Sampling data:** It means the later calculations are for the sampling data, not testing data.

**15. Test data:** It means the later calculations are for the testing data, not sampling data.

**16. Use IO model:** If the user activates this checkbox, after the program predicted the sampling data, it will remove the samples that are out of the screens range.

**17. Get pixels:** If the user activates this checkbox, the program will predict the sampling or testing data.

**18. See pixels:** If the user activates this checkbox, the program will show the locations that the user has looked.

**19. Get fixations:** The program will calculate the fixations of the user by three parameters of 20, 21, and 23.

**20. ST:** Abbreviation for saccade threshold. To separate fixations, it is needed to firstly calculate the velocity of eye movement. So, by putting a threshold on the velocity, it's possible to compute the moments that the user changed their viewpoint.

**21. DFT:** Abbreviation for discard fixation time. Fixations that last less than this time, will be removed.

**22. MFR:** Abbreviation for merged fixations ratio. Fixations that are close together, will be added. Two numbers are for two directions.

**23. Start:** This button starts the program. If you activate one of the checkboxes, the program just do that specific one. If you select some of the checkboxes, the program goes for all of the selected ones, one after another.

**24. Stop:** This button stops the program in every step that it's running.
