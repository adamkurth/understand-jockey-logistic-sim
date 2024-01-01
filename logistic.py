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
        
    def update_model(self, positions, speeds):
        # prepare featuress for each horse compared to other horses
        num_features_per_horse = 2 * (self.num_horses - 1)
        self.features = np.zeros((self.num_horses, num_features_per_horse))

        for i in range(self.num_horses):
            features = []
            for j in range(self.num_horses):
                if i != j:
                    features.extend([(positions[i] - positions[j]), (speeds[i] - speeds[j])])
            
            self.features[i] = features
        
        # determine the leading horse
        leading_horse = np.argmin(positions)
        self.target = np.full(self.num_horses, 0)  # Initialize all as non-leaders
        self.target[leading_horse] = 1  # Leading horse
            
        # fit the model
        if len(np.unique(self.target)) >= 2:
            self.model.fit(self.features, self.target)
            self.model_fitted = True
        else:
            if not self.model_fitted:
                print("Not enough class variation for training. Assigning random targets for initial diversity.")
                self.target = np.random.randint(0, 2, self.num_horses)
                
    def predict_probabilities(self):
        if self.model_fitted:
            return self.model.predict_proba(self.features)
        else:
            print("Model not fitted yet. Can't predict probabilities.")
            return np.full((self.num_horses, 2), 1 / self.num_horses)
        