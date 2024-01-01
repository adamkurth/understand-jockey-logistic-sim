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
variation = 0.0005 # small variations
horse_speeds = np.abs(np.random.normal(loc=base_speed, scale=variation, size=num_horses)) # positive
horse_positions = np.zeros(num_horses) #initialize positions

# animation
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1, 1)
ax.set_aspect('equal') # ensure no distortion

fig_leaderboard, ax_leaderboard = plt.subplots()
leaderboard_text = ax_leaderboard.text(0.05, 0.95, '', verticalalignment='top')
ax_leaderboard.axis('off')

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
    global horse_positions, horse_speeds, leaderboard_text

    # Update horse positions on the circular track
    horse_positions = (horse_positions + horse_speeds) % (2 * np.pi)

    # Update the race model and predict probabilities
    race_model.update_model(horse_positions, horse_speeds)
    probabilities = race_model.predict_probabilities()

    # Update leaderboard with sorted probabilities
    sorted_indices = np.argsort(-probabilities[:, 1])  # Sort horses by probability of leading
    leaderboard = "Leaderboard:\n"
    for i, idx in enumerate(sorted_indices):
        prob = probabilities[idx, 1] if probabilities is not None else "N/A"
        leaderboard += f"Rank {i+1}: Horse {idx} - {prob:.4f}\n"
    leaderboard_text.set_text(leaderboard)

    # Update visuals for each horse on the race track
    for i, point in enumerate(points):
        x, y = track(horse_positions[i])
        point.set_data([x], [y])

    fig_leaderboard.canvas.draw()


    # determine if leading horse has completed the number of laps.
    leading_horse_dist = np.max(horse_positions)
    if leading_horse_dist >= num_laps * 2 * np.pi:
        plt.close('all')
        
    return points

ani = FuncAnimation(fig, update, frames=np.linspace(0, race_dist, 1200), init_func=init, blit=True, interval=10)
plt.show()
