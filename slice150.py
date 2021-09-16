import matplotlib.pyplot as plt
import numpy as np
import math
import collections

file_path = "slice150.raw"
size = 512
scale = 255

student = "Dipinto Maria Carmela"

#read 16-bit data from file
arr = np.fromfile(file_path, dtype=np.uint16)
#total length of array
totalNumberofValues = len(arr)
min = min(arr)
max = max(arr)

# obtain matrix 512x512 to obtain a bidimensional matrix
arr = arr.reshape(size, size)

# linear transformation function
def linear(value):
    return (value - min) / (max - min) * scale
	
# nolinear transformation function
def nonlinear(value):
    log_min = math.log(min) if min > 0 else 0
    log_max = math.log(max) if max > 0 else 0
    if value > 0:
        log = math.log(value)
        if log > 0:
            return (log - log_min) / (log_max - log_min) * scale
    return 0
	
def calculateVariance(mean, arr):
	sd = 0
	for row in arr:
		for col in row:
			sd += (col - mean)**2
	variance = sd/totalNumberofValues
	return variance	
	
# Process dataset and generate data for the next steps. I insert this step here to not doing it multiple times in each of the next steps.
totalValue = 0                          #sum of values for mean value(see the task B)
histogram = {}                          #store data for histogram (see the task C)
arr_linear = np.zeros((size, size))     #array for linear transformation (see the task D)
arr_nonlinear = np.zeros((size, size))  #array for non-linear transformation (see the task E)
for x in range(0, size):
	for y in range(0, size):
		value = arr[x, y]
		#add value to totalValue
		totalValue += value
		# linear transformation
		arr_linear[x, y] = linear(arr[x, y])  
		# non-linear transformation
		arr_nonlinear[x, y] = nonlinear(arr[x, y])  		
        #increment histogram  key value if exists else add it to dict
		if value in histogram:
			histogram[value] += 1
		else:
			histogram[value] = 1

#init: display slice150 image
#plt.clf()
#plt.subplots_adjust(bottom=0.2)
#plt.figtext(0.99, 0.01, student, wrap=True,
#			ha="right", va="bottom", fontsize=8)
#plt.title("Slice150 2D dataset image")
#plt.imshow(arr, 'gray', interpolation='nearest', origin='lower')
#plt.savefig("./outputs/slice150.png")
#plt.show()

#####################################################################	
## (a) Draw a profile line through line 256 of this 2D data set.    
#####################################################################
profile_line = []
y = 255
for x in range(0, size):
    profile_line.append(arr[x, y])
	

plt.clf()
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.99, 0.01, student, wrap=True,
			ha="right", va="bottom", fontsize=8)
plt.title("Profile line through line 256")
plt.plot(profile_line)
plt.xlabel('r (Pixel Position along Profile Line) ')
plt.ylabel('s=T(r)')         
plt.savefig("./profile_line_slice150.png")
plt.show

############################################################################
# (b) Calculate the mean value and the variance value of this 2D data set. 
############################################################################
mean = totalValue/totalNumberofValues
variance = calculateVariance(mean, arr)  


with open("./mean_and_variance.txt", "w+") as out:
    out.write("Mean value is: " +str(mean))
    out.write("\n\nVariance value is: " +str(variance))
###################################################################################################################
# (c) Display a histogram of this 2D data set (instead of bars you may use a line graph to link occurrences along 
# the x-axis).                                                                                                    
###################################################################################################################

plt.clf()
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.99, 0.01, student, wrap=True,
			ha="right", va="bottom", fontsize=8)
plt.title("Slice150th Dataset Histogram")
plt.bar(histogram.keys(), histogram.values(),width=0.6)
bins = np.linspace(min,max+1,max+1)
plt.xlabel('Values')
plt.ylabel('Number of occurrences')
plt.savefig("./histrogram_slice150.png")
	


#####################################################################
#d: Rescaling using Linear Transformation
#####################################################################
plt.clf()
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.99, 0.01, student, wrap=True,
			ha="right", va="bottom", fontsize=8)
plt.title("Rescaling using Linear Transformation")
plt.imshow(arr_linear, 'gray', interpolation='nearest', origin='lower')
plt.savefig("./linear_slice150.png")
plt.show()

#####################################################################
#e: Rescaling using non Linear Transformation
#####################################################################
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.99, 0.01, student, wrap=True,
			ha="right", va="bottom", fontsize=8)
plt.title("Rescaling using non Linear Transformation")
plt.imshow(arr_nonlinear, 'gray', interpolation='nearest', origin='lower')
plt.savefig("./nonLinear_slice150.png")
plt.show()

#####################################################################
#f: 11*11 Box Car Smoothing
#####################################################################
def boxCarMatrix(arr, i, j, boxCarFilter):
	sum = 0
	for row in range(i,i+11):
		for col in range(j, j+11):
			sum = sum + arr[row][col]
	retVal = len(boxCarFilter) * len(boxCarFilter)
	return (sum/retVal)

def visualiseBoxCarFilteredData(arr, boxCarFilter, outMatrix):
	
	for i in range(0,len(arr)):
		#checking for row limit
		if((len(arr)-i)>(len(boxCarFilter)-1)):
			for j in range(0,len(arr[i])):
				if((len(arr[i])-j)>(len(boxCarFilter)-1)):
					midVal = boxCarMatrix(arr, i , j, boxCarFilter)
					outMatrix[i][j] = midVal
				
boxCarFilter = np.zeros((11,11))
outMatrix = np.zeros((512,512)) 
visualiseBoxCarFilteredData(arr,boxCarFilter,outMatrix)
plt.title('Box Car Smoothing Filter')
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.99, 0.01, student, wrap=True,
			ha="right", va="bottom", fontsize=8)
plt.imshow(outMatrix, 'gray', origin='lower')
plt.savefig("./boxcar_smoothing_slice150.png")
plt.show()

#####################################################################
#g: 11*11 Median Filter
#####################################################################
def medianFilterMatrix(arr, i, j):
	#sort the matrix to replace it with the median value 
	medianList = []
	for row in range(i, i+11):
		for col in range(j, j+11):
			medianList.append(arr[row][col])
	medianList.sort()
	return medianList[int(len(medianList)/2)]

def visualiseMedianFilteredData(arr, medianFilter, medMatrix):
	
	for i in range(0,len(arr)):
		#checking for row limit
		if((len(arr)-i)>(len(medianFilter)-1)):
			for j in range(0,len(arr[i])):
				if((len(arr[i])-j)>(len(medianFilter)-1)):
					midVal = medianFilterMatrix(arr, i , j)
					medMatrix[i][j] = midVal
				
medianFilter = np.zeros((11,11))
medMatrix = np.zeros((512,512)) 
visualiseMedianFilteredData(arr,medianFilter,medMatrix)
plt.title('Median Filter')
#plt.subplots_adjust(bottom=0.2)
plt.figtext(0.99, 0.01, student, wrap=True,
			ha="right", va="bottom", fontsize=8)
plt.imshow(medMatrix, 'gray', origin='lower')
plt.savefig("./median_filter_slice150.png")
plt.show()

######################################################################
