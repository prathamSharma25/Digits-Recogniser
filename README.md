# Digits-Recogniser
MACHINE LEARNING MODEL TO PREDICT HAND-DRAWN DIGITS IN PYTHON.

This model identifies the hand-drawn digit based on the data about each pixel of the image of the hand-drawn digit. Digits from 0 through 9 are hand-drawn.

The data files train.csv and test.csv contain pixel-values for gray-scale images of hand-drawn digits, from 0 through 9.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

The pixels make up the image like this:
000 001 002 003 .... 026 027

028 029 030 031 .... 054 055

056 057 058 059 .... 082 083

 :   :   :   :  ....  :   :
 
 :   :   :   :  ....  :   :
 
728 729 730 731 .... 754 755

756 757 758 759 .... 782 783 

The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column. Submission file contains "ImageID" and "Digit" for aech of the 28000 images in the test dataset. 

Logistic Regression algorithm is used here to train the ML model to predict the hand-drawn digit. data is classified into 10 classes, one for each digit from 0 through 9.
