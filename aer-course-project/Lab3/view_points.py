import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


# Function to create 3D arrows
def Arrow3D(ax, x, y, z, dx, dy, dz, color):
    ax.quiver(x, y, z, dx, dy, dz, color=color, arrow_length_ratio=0.1)


# Read the CSV file into a DataFrame
df = pd.read_csv("lab3_pose.csv")

# Extract position and orientation data
positions = df[["p_x", "p_y", "p_z"]].values
quaternions = df[["q_w", "q_x", "q_y", "q_z"]].values

# Update interval
interval = 10

# Initialize the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Set axis limits
ax.set_xlim([np.min(positions[:, 0]), np.max(positions[:, 0])])
ax.set_ylim([np.min(positions[:, 1]), np.max(positions[:, 1])])
ax.set_zlim([np.min(positions[:, 2]), np.max(positions[:, 2])])


# Initialize plot objects
(points,) = ax.plot([], [], [], "bo")
arrows = []


# Update function for animation
def update(frame):
    # print("image_", frame)
    # Clear previous arrows
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 3])

    # Update point positions
    (points,) = ax.plot([], [], [], "bo")
    points.set_data(positions[frame, 0], positions[frame, 1])
    points.set_3d_properties(positions[frame, 2])

    # Update coordinate system arrows
    rotation_matrix = Rotation.from_quat(quaternions[frame]).as_matrix()
    colors = ["r", "g", "b"]
    for i in range(3):
        Arrow3D(
            ax,
            positions[frame, 0],
            positions[frame, 1],
            positions[frame, 2],
            rotation_matrix[0, i],
            rotation_matrix[1, i],
            rotation_matrix[2, i],
            color=colors[i],
        )
    # row_data = df.iloc[frame]
    # position_info = f'Position: {row_data["p_x"]}, {row_data["p_y"]}, {row_data["p_z"]}'
    # quaternion_info = f'Quaternions: {row_data["q_w"]}, {row_data["q_x"]}, {row_data["q_y"]}, {row_data["q_z"]}'
    # ax.text2D(0.05, 0.95, f'Frame: {frame}, {position_info}, {quaternion_info}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text2D(
        0.05,
        0.95,
        f"Frame: {frame}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )


# Create animation
ani = FuncAnimation(fig, update, frames=len(df), interval=interval)

# Show plot
plt.show()
