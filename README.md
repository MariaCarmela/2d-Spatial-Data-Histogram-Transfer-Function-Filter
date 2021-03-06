# 2d-Spatial-Data-Histogram-Transfer-Function-Filter
Exercise on 2d Spatial Data, Histogram, Transfer Function, Filter

This is an exercise on how to manipulate 2d Spatial Data. 
# Installation
Please install needed packages:

pip install numpy
pip install matplotlib
# How to run the code
You can run the project in this way:

python slice150.py

# Dataset
The data consider is one slice of a CT angiographic scan. You can find it in the repository (slice150.raw). Data set and descriptions
(TermsOfUse_slice150.txt and DataCharacteristics_slice150.txt) are provided  as well.
Data is stored as 16-bit integer values.
# Tasks
(a) Draw a profile line through line 256 of this 2D data set.

(b) Calculate the mean value and the variance value of this 2D data set.

(c) Display a histogram of this 2D data set (instead of bars you may use a line graph to link occurrences along the x-axis).

(d) Rescale values to range between 0 and 255 using a linear transformation.

(e) Rescale values to range between 0 and 255 using a different (e.g. non-linear) transformation.

(f) Use an 11x11 boxcar smoothing filter on the 2D data set.

(g) Use an 11x11 median filter on the 2D data set.

# Output
The output for the subtasks for (a) is profile_line_slice150.png

The output for the subtasks for (b) is mean_and_variance.txt  

The output for the subtasks for (c) is histrogram_slice150.png

The output for the subtasks for (d) is linear_slice150.png

The output for the subtasks for (e) is nonLinear_slice150.png

The output for the subtasks for (f) is boxcar_smoothing_slice150.png

The output for the subtasks for (g) is median_filter_slice150.png


