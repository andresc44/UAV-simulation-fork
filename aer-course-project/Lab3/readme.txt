To run the code for AER1216 lab 3, please run 'lab3_main.py' main method.

This will plot and displace three plots with the sanitized locations of the markers using 3 iterations of k-means clustering,
and will save the processed images to /processed_frames folder directory.

The main method is broken down into two parts, the first part, the for loop. This is where we do image processing
to identify the target on the image frame, and then use the drone state and camera parameters to find the marker's centroid location.
This for loop can be run on realtime as we acquire image frames and measure the drone state.

The second part uses the location found on all frames for all targets, and use k-means clustering to sanitized and clearn these location to end up with
six locations for the six different markers. These six locations are then printed to the console.

We created helper methods to modularize the code. There are 6 methods for k-means clustering steps,
one method for getting target location form drone state and frame image, and one method to get the state of the frame from the state dataframe object.

At the top of the lab3_main.py file, we have parameters to tune the image-processing, k-means clustering, and target format.
We also specify the input/output directories there. There is also a parameter that specify whether to save the processed images or not.