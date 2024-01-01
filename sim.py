import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define constants
num_horses = 10
num_laps = 3
race_dist = num_laps * 2 * np.pi  # 2pi * radius

# Assumption: each hprse has base speed, vary closely to the mean.
base_horse_speed = np.random.normal(loc=0.015, scale=0.001, size=num_horses)
base_horse_speed = np.abs(base_horse_speed) # positive

# initialize angle
horse_angle = np.zeros(num_horses)

# animation
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1, 1)
ax.set_aspect('equal') # ensure no distortion

# markers
colors = plt.cm.rainbow(np.linspace(0, 1, num_horses))
points = [ax.plots([], [], marker='o', color=colors[i])[0] for c in range(num_horses)]

def track(angle):
    """Calculate the x,y coordinates for an angle on the track"""
    a = 1.0 # semi-major axis (x-radius)
    b = 0.5 # semi-minor axis (y-radius)
    x = a * np.cos(angle)
    y = b * np.sin(angle)
    return x, y

def init():
    """Initialize the animation"""
    for point in points:
        point.set_data([], [])
    return points

def update(frame):
    global horse_angle
    horse_angle += base_horse_speed
    # update positions
    for i in range(num_horses):
        angle = horse_angle[i]
        x, y = track(angle)
        points[i].set_data(x, y)
    return points

ani = FuncAnimation(fig, update, frames=np.linspace(0, race_dist, 1200), init_func=init, blit=True, interval=10)
plt.show()
