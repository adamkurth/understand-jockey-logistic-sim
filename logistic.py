import numpy as np
from sklearn.linear_model import LogisticRegression

class RaceModel:
    def __init__(self, num_horses):
        # initialize logit model
        self.model = LogisticRegression()
        self.num_horses = num_horses
        self.features = None
        self.target = None
        self.model_fitted = False
        
    def calculate_remaining_distances(self, positions, race_length=2 * np.pi):
        """Calculate the remaining distance for each horse to the finish line."""
        # Assuming positions are given as the angle on a circular track (0 to 2π)
        # The finish line corresponds to 2π (complete a lap)
        remaining_distances = (race_length - positions) % race_length
        return remaining_distances
    
    def update_model(self, positions, speeds):
        """Update the model with the current positions and speeds of the horses."""
        # Calculate features
        remaining_distances = self.calculate_remaining_distances(positions)
        self.features = np.vstack((remaining_distances, speeds)).T

        # Determine the target (which horse is leading)
        # This is a simple way to assign the target where the horse with the least distance to finish is considered leading
        self.target = np.zeros(self.num_horses)
        self.target[np.argmin(remaining_distances)] = 1

        # enough class variation => fit the model
        if len(np.unique(self.target)) >= 2:
            self.model.fit(self.features, self.target)
            self.model_fitted = True
                
    def predict_probabilities(self):
        """Predict the winning probabilities for each horse."""
        if self.model_fitted:
            return self.model.predict_proba(self.features)
        else:
            return None
        