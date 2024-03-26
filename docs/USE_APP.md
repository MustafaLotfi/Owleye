## This document shows you how to use Owleye application

When you run the program, a window like the one in below will be appeared:

![Screenshot 2024-03-25 142116](https://github.com/MustafaLotfi/Owleye/assets/53625380/1d311246-4273-4092-a1ab-d7dd2b84173f)

Also, after running, a folder called "subjects" will be created. In this folder, a folder will be created for each subject based on the subject number in the UI. Then, you can adjust the items that you need, and finally "start" the program. In following, the items will be explained.

**1. Subject number:** A specific number that you enter as the ID of the subject.

**2. Camera ID:** Usually it is 0, but you can try other numbers if you have several webcams on your system.

**3. Camera:** by selecting this item, the webcam will be turned on. So you can see your picture and the landmarks that already are detected on your face.

Calibration: By selecting this item and clicking on the "start" button, the program will be ready to collect data (inputs and outputs of the models) from the user. So, a white point will be appeared on the screen. As soon as you press the "SPACE" key on keyboard, the program starts collecting data for a particlular time. While this, the background bacome black and the user should look at the white point during this time. If the data collection ends, the screen will become gray and the point goes to another location. The user can look everywhere and actually rest in this situation. Again, the "SPACE" key should be pressed for data collection and all of the explained process be repeated.

Subject name: This item is arbitrary. you can enter the user's name.

Describtion: This section is arbitrary too. you can enter any information that your user has.

Calibration grid: This item can have three types of integer numbers.

- 2 numbers (n, c): The white point starts to move just horizontally in the screen in n rows. Each row contains c locations that the point goes through.
- 3 numbers (n, m , c): The white point does not move. just in a grid by size of n x m (like a matrix) remains fixed in each location.
- 4 numbers (n, c, m, d): The white point moves both horizontally and vertically, in n rows with c locations and m columns with d locations.



