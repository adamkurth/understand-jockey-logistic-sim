---
runme:
  id: 01HK3JJKRMHBBEVF67TH422A1N
  version: v2.0
---

# Logistic Regression in Horse Racing Prediction

### Introduction

Logistic Regression is a statistical method for analyzig data in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomus variable (pass/fail, win/lose). In the context of horse racing, logistic regression can be used to predict the probability of a particular horse winning. 

### Fundamental Concepts

The logitstic function (i.e. sigmoid) is an S-shaped curve: $∀_{x \in \mathbb{R}} : x \in [0,1]$
$$σ(z) = \frac{1}{1+e^{-z}}$$ 

Such that:
- $σ(z)$ is the output
- $z$ is the input to the function (linear combination of weights and obs. variables).

### Model Representation

In logistic regression, typically trying to model the probability that a given input point belongs to certain cases (horse wins). The probability that an outcome will occur is modeled as a linear function of combination of predictor variables.

$$P(Y=1 | X) = \frac{1}{1+\exp(-(β_0 + \beta_1X_1 + ⋯ + \beta_nX_n))}$$

Such that: 

- $P(Y=1 | X)$ is the prob. that the horse wins given predictors. 
- $X_1, X_2, X_3 ⋯ X_n$ are the features (speed, distance to finish).
- $\beta_0, \beta_1, \cdots \beta_n$ are the coefficients to be learned from the training data. 

### Learning the Model

The coefficients $\beta_0, \beta_1, \cdots, \beta_n$ are usually learned through a process called Maximum Liklihood Estimation (MLE). The idea is to find the set of parameters that make the observed data most probable. 

## Application to Horse Racing

Feature Engineering: Some might include:
- *Distance to finish*: remaining distance each horse has to cover to finish the race. 
- *Speed*: Current speed or average speed of the horse. 
  
### Model Fitting 

Given a set of race data: 
- **X**: A matrix where each row represents a horse, each column represents a feature of that horse.
- **Y**: A vector where each entry represents whether or not the horse is in the lead or not (1 or 0).
  
### Prediction
Once the model is fitted, it can predict the winning probability for each horse in a new race. The model will output a probability score between 0 and 1 for each horse, indicating the likelihood of that horse winning the race.

### Interpretation
- *Coefficients*: Positive coefficients increase the log-odds of the response variable (∴ increase the probability) and negative coefficients decrease the log-odds (∴ decrease the probability).
- *Odds Ratio*: For one-unit increase in the predictor variable, the odds ratio is the factor by which the odds of the outcome increase, or decrease given the coefficient. 
