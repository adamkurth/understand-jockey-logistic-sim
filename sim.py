import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from logistic import RaceModel

# Define constants
num_horses = 10
num_laps = 3
race_dist = num_laps * 2 * np.pi  # 2pi * radius

# logit model
race_model = RaceModel(num_horses)

# Assumption: each hprse has base speed, vary closely to the mean.
base_speed = 0.015 # for all horses
variation = 0.0005 # small variation f
horse_speeds = np.abs(np.random.normal(loc=base_speed, scale=variation, size=num_horses)) # positive
horse_positions = np.zeros(num_horses) #initialize positions

# animation
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1, 1)
ax.set_aspect('equal') # ensure no distortion

# text animaition for diplaying window likihoods 
prob_text = ax.text(0.5, 1.05, '', transform=ax.transAxes)

# markers
colors = plt.cm.rainbow(np.linspace(0, 1, num_horses))
points = [ax.plot([], [], 'o', color=colors[i])[0] for i in range(num_horses)]
prob_texts = [ax.text(0.5, 1.05 - 0.05 * i, '', transform=ax.transAxes) for i in range(num_horses)]

def track(angle):
    """Calculate the x,y coordinates for an angle on the track"""
    a = 1.0 # semi-major axis (x-radius)
    b = 0.5 # semi-minor axis (y-radius)
    return a * np.cos(angle), b * np.sin(angle)

def init():
    """Initialize the animation"""
    for point in points:
        point.set_data([], [])
    return points

def update(frame):
    global horse_positions, horse_speeds
    
    # Update positions based on speeds
    horse_positions += horse_speeds
    
    # Update the model and get the probabilities
    race_model.update_model(horse_positions, horse_speeds)
    probabilities = race_model.predict_probabilities()
    
    # Update the text annotations with the probabilities
    if probabilities is not None and probabilities.shape[1] > 1:
        for i, prob_text in enumerate(prob_texts):
                prob_text.set_text(f"Horse {i}: {probabilities[i][1]:.4f}")

    # Update the positions of the points representing the horses
    for i in range(num_horses):
        x, y = track(horse_positions[i])
        points[i].set_data([x], [y])

    return points + prob_texts

ani = FuncAnimation(fig, update, frames=np.linspace(0, race_dist, 1200), init_func=init, blit=True, interval=10)
plt.show()
