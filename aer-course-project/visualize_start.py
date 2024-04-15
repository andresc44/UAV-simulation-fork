import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given points
# points = np.array([[-1.1, -3.1, -0.2],
#                    [-1., -3.,  0.1],
#                    [ 0.2, -2.5, 1. ],
#                    [ 0.5, -2.4, 1.1 ]])

points = np.array([[-0.5,  1.6,  1. ],
                    [-0.5,  2.,   0.1]])

# Step size for the additional points
step = 1.5

# Add additional points
extra_point1 = np.array([points[0,0], points[0,1]-step, points[0,2]])
extra_point2 = np.array([points[1,0], points[1,1], points[1,2]-step])

points = np.vstack((extra_point1, points, extra_point2))
# Extract x, y, and z coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Fit 4th-degree polynomial to each dimension
coefficients_x = np.polyfit(np.arange(len(points)), x, 4)
coefficients_y = np.polyfit(np.arange(len(points)), y, 4)
coefficients_z = np.polyfit(np.arange(len(points)), z, 4)

# Generate points for the curve
duration = 10
dt = 1/ 30.0
steps = int(duration // dt)

t = np.linspace(1, 2, steps)
x_interp = np.polyval(coefficients_x, t)
y_interp = np.polyval(coefficients_y, t)
z_interp = np.polyval(coefficients_z, t)

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='red', label='Points')

# Plot the curve
ax.plot(x_interp, y_interp, z_interp, color='blue', label='4th-degree Polynomial')

ax.set_xlim([-1, 0])  # Set x-axis limits
ax.set_ylim([1, 2])  # Set y-axis limits
ax.set_zlim([0, 1.5])  # Set z-axis limits

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('4th-degree Polynomial Interpolation in 3D')
ax.legend()
plt.show()
