**# Generalized Linear Models for Race Simulation: Technical Specification

## Table of Contents
1. [Quick Start](#quick-start)
2. [Introduction](#introduction)
3. [Binary Logistic Regression](#binary-logistic-regression)
4. [Multinomial Logistic Regression](#multinomial-logistic-regression)
5. [Conditional Logistic Regression](#conditional-logistic-regression)
6. [Estimation Theory](#estimation-theory)
7. [Model Diagnostics](#model-diagnostics)
8. [Implementation Considerations](#implementation-considerations)

---

## Quick Start

### Running the Simulation

This project implements a horse race simulation with real-time win probability estimation using GLM/IRLS (Iteratively Reweighted Least Squares).

#### Prerequisites

Install the required R packages:

```r
install.packages(c("shiny", "ggplot2", "dplyr", "tidyr", "scales"))
```

#### Launch the Application

```bash
# From the terminal
cd /path/to/understand-jockey-logistic-sim
Rscript -e "shiny::runApp('app.R')"
```

Or from within R/RStudio:

```r
setwd("/path/to/understand-jockey-logistic-sim")
shiny::runApp("app.R")
```

### Project Structure

| File | Description |
|------|-------------|
| `logistic.r` | Core GLM/IRLS engine implementing binary, multinomial, and conditional logistic regression |
| `app.R` | Shiny application with animated race visualization |
| `README.md` | This technical specification document |

### Features

- **Real-time IRLS computation** — Watch the Newton-Raphson iterations converge
- **Multiple model types**:
  - Binary logistic (lead horse probability)
  - Multinomial logit (softmax over all horses)
  - Conditional logit (McFadden's choice model)
- **Animated race track** — Horses move on an elliptical track
- **Live probability updates** — Bar chart showing win probabilities
- **Mathematical output** — View coefficients, standard errors, deviance, and more

---

## Introduction

This document provides the mathematical foundation for implementing Generalized Linear Models (GLMs) in the context of horse race simulation. The goal is to replace black-box estimators with transparent, analytically-derived methods that align with the GLM framework presented in:

- **Agresti (2013)** — *Categorical Data Analysis*
- **McCullagh & Nelder (1989)** — *Generalized Linear Models*
- **Course materials** — PHP2605 Lecture Series

---

## Binary Logistic Regression

### The Exponential Family Foundation

The GLM framework requires a response distribution from the exponential family. For binary outcomes $Y_i \in \{0, 1\}$, we use the Bernoulli distribution.

#### Bernoulli as Exponential Family

The probability mass function:

$$f(y_i; \pi_i) = \pi_i^{y_i}(1 - \pi_i)^{1 - y_i}$$

Can be written in exponential family form:

$$f(y_i; \theta_i) = \exp\left(y_i \theta_i - \log(1 + e^{\theta_i}) \right)$$

where the **natural parameter** is:

$$\theta_i = \log \left(\frac{\pi_i}{1 - \pi_i}\right)$$

This reveals why the logit is the **canonical link function** — it directly maps the linear predictor to the natural parameter.

### The Three Components of a GLM

#### 1. Random Component

$$Y_i \stackrel{\text{ind}}{\sim} \text{Bernoulli}(\pi_i), \quad i = 1, \ldots, n$$

Properties:
- $\mathbb{E}[Y_i] = \pi_i$
- $\text{Var}(Y_i) = \pi_i(1 - \pi_i)$

#### 2. Systematic Component (Linear Predictor)

$$\eta_i = \mathbf{x}_i^\top \boldsymbol{\beta} = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip}$$

In matrix form:

$$\boldsymbol{\eta} = \mathbf{X}\boldsymbol{\beta}$$

where $\mathbf{X}$ is the $n \times (p+1)$ design matrix including the intercept column.

#### 3. Link Function

The **logit link** (canonical for Bernoulli):

$$g(\pi_i) = \log\left(\frac{\pi_i}{1 - \pi_i}\right) = \eta_i$$

The **inverse link** (response function):

$$\pi_i = g^{-1}(\eta_i) = \frac{\exp(\eta_i)}{1 + \exp(\eta_i)} = \frac{1}{1 + \exp(-\eta_i)}$$

This is the **logistic function**, often denoted $\sigma(\eta_i)$ or $\text{expit}(\eta_i)$.

### Alternative Link Functions

| Link | Function $g(\pi)$ | Inverse $g^{-1}(\eta)$ | Use Case |
|------|-------------------|------------------------|----------|
| Logit | $\log\frac{\pi}{1-\pi}$ | $\frac{e^\eta}{1+e^\eta}$ | Default, symmetric |
| Probit | $\Phi^{-1}(\pi)$ | $\Phi(\eta)$ | Latent normal threshold |
| Complementary log-log | $\log(-\log(1-\pi))$ | $1 - \exp(-e^\eta)$ | Asymmetric, rare events |
| Cauchit | $\tan(\pi(\pi - 0.5))$ | — | Heavy tails |

### Interpretation of Coefficients

For a unit increase in $x_j$, holding other covariates constant:

$$\log\left(\frac{\pi}{1-\pi}\right) \to \log\left(\frac{\pi}{1-\pi}\right) + \beta_j$$

Thus:

$$\text{Odds Ratio} = \frac{\text{Odds}_{\text{new}}}{\text{Odds}_{\text{old}}} = e^{\beta_j}$$

**Key insight**: $\beta_j$ represents the change in **log-odds** per unit change in $x_j$.

---

## Multinomial Logistic Regression

For horse racing with $J$ horses, the natural extension is multinomial logistic regression (also called **polytomous logistic regression** or **softmax regression**).

### Setup

Let $Y_i \in \{1, 2, \ldots, J\}$ denote the winner of race $i$, where $J$ is the number of horses.

Define:

$$\pi_{ij} = P(Y_i = j) = P(\text{Horse } j \text{ wins race } i)$$

Constraint: $\sum_{j=1}^{J} \pi_{ij} = 1$

### Baseline-Category Logit Model (Agresti Ch. 8)

Choose category $J$ as the **reference category**. Model the log-odds relative to the baseline:

$$\log\left(\frac{\pi_{ij}}{\pi_{iJ}}\right) = \mathbf{x}_i^T \boldsymbol{\beta}_j, \quad j = 1, \ldots, J-1$$

This gives $J-1$ equations with $J-1$ coefficient vectors $\boldsymbol{\beta}_1, \ldots, \boldsymbol{\beta}_{J-1}$.

### Solving for Probabilities

From the log-odds equations:

$$\frac{\pi_{ij}}{\pi_{iJ}} = \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_j)$$

Using the constraint $\sum_{j=1}^J \pi_{ij} = 1$:

$$\pi_{iJ} + \sum_{j=1}^{J-1} \pi_{iJ} \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_j) = 1$$

$$\pi_{iJ} \left(1 + \sum_{j=1}^{J-1} \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_j)\right) = 1$$

Therefore:

$$\pi_{iJ} = \frac{1}{1 + \sum_{k=1}^{J-1} \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_k)}$$

$$\pi_{ij} = \frac{\exp(\mathbf{x}_i^\top \boldsymbol{\beta}_j)}{1 + \sum_{k=1}^{J-1} \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_k)}, \quad j = 1, \ldots, J-1$$

### Expit Formulation

Equivalently, setting $\boldsymbol{\beta}_J = \mathbf{0}$ (identifiability constraint):

$$\pi_{ij} = \frac{\exp(\mathbf{x}_i^\top \boldsymbol{\beta}_j)}{\sum_{k=1}^{J} \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_k)}, \quad j = 1, \ldots, J$$

This is the **expit function**, which maps $J$ linear predictors to a probability simplex.

### Multinomial Distribution

For a single race with one observation per race (exactly one winner):

$$Y_i \mid \boldsymbol{\pi}_i  \sim \text{Multinomial}(1, \boldsymbol{\pi}_i)$$

where $\boldsymbol{\pi}_i = (\pi_{i1}, \ldots, \pi_{iJ})^\top$.

The probability mass function:

$$P(\mathbf{y}_i \mid\boldsymbol{\pi}_i ) = \frac{1!}{\prod_{j=1}^{J} y_{ij}!} \prod_{j=1}^{J} \pi_{ij}^{y_{ij}} = \prod_{j=1}^{J} \pi_{ij}^{y_{ij}}$$

where $y_{ij} = \mathbb{1}(Y_i = j)$ is the indicator that horse $j$ won race $i$.

### Log-Likelihood for Multinomial Logistic Regression

Then $\mathbf{y}_1, \ldots, \mathbf{y}_n$ are independent with respective $\boldsymbol{\pi}_1, \ldots, \boldsymbol{\pi}_n$. Then for $n$ independent races that is, the log-likelihood is:

$$\ell(\boldsymbol{\beta} \mid \mathbf{y}_1, \ldots, \mathbf{y}_n, \boldsymbol{\pi}_1, \ldots, \boldsymbol{\pi}_n) = \ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \sum_{j=1}^{J} y_{ij} \log(\pi_{ij})$$

Substituting the expit form of $\pi_{ij}$:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \sum_{j=1}^{J} y_{ij} \left[ \mathbf{x}_i^\top \boldsymbol{\beta}_j - \log\left(\sum_{k=1}^{J} \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_k)\right) \right]$$

Since $\sum_j y_{ij} = 1$:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ \sum_{j=1}^{J} y_{ij} \mathbf{x}_i^\top \boldsymbol{\beta}_j - \log\left(\sum_{k=1}^{J} \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_k)\right) \right]$$

### Score Equations

Define $\boldsymbol{\beta} = (\boldsymbol{\beta}_1^\top, \ldots, \boldsymbol{\beta}_{J-1}^\top)^\top$ as the stacked parameter vector.

The score function with respect to $\boldsymbol{\beta}_j$:

$$\frac{\partial \ell}{\partial \boldsymbol{\beta}_j} = \sum_{i=1}^{n} (y_{ij} - \pi_{ij}) \mathbf{x}_i = \mathbf{X}^\top(\mathbf{y}_j - \boldsymbol{\pi}_j)$$

where $\mathbf{y}_j = (y_{1j}, \ldots, y_{nj})^\top$ and $\boldsymbol{\pi}_j = (\pi_{1j}, \ldots, \pi_{nj})^\top$.

Setting to zero gives:

$$\mathbf{X}^\top \mathbf{y}_j = \mathbf{X}^\top \boldsymbol{\pi}_j, \quad j = 1, \ldots, J-1$$

### Hessian Matrix

The second derivatives form the observed information matrix. For $j, k \in \{1, \ldots, J-1\}$:

$$ I_T(Y, \boldsymbol{\beta}) = \frac{\partial^2 \ell}{\partial \boldsymbol{\beta}_j \partial \boldsymbol{\beta}_k^\top} = -\sum_{i=1}^{n} \pi_{ij}(\delta_{jk} - \pi_{ik}) \mathbf{x}_i \mathbf{x}_i^\top$$

where $\delta_{jk}$ is the Kronecker delta (1 if $j=k$, 0 otherwise).

In block matrix form:

$$\mathbf{H} = -\mathbf{X}^\top \mathbf{W} \mathbf{X}$$

where $\mathbf{W}$ is a block-diagonal weight matrix with blocks:

$$\mathbf{W}_{j,k} = \text{diag}(\pi_{ij}(\delta_{jk} - \pi_{ik}))$$

### Fisher Information

The **Fisher information** is defined as the negative expected value of the Hessian:

$$\mathcal{I}(\boldsymbol{\beta}) = -\mathbb{E}[\mathbf{H}] = \mathbb{E}\left[\mathbf{X}^\top \mathbf{W} \mathbf{X}\right]$$

For multinomial logistic regression, the weight matrix $\mathbf{W}$ is a function of $\boldsymbol{\pi}$, which depends on $\boldsymbol{\beta}$ deterministically through the expit function. Since $\mathbf{X}$ is fixed (non-random design) and $\mathbf{W}$ depends only on the parameters (not on the random response $\mathbf{Y}$), we have:

$$\mathcal{I}(\boldsymbol{\beta}) = \mathbf{X}^\top \mathbf{W} \mathbf{X}$$

where the weight matrix $\mathbf{W}$ encodes the **variance-covariance structure** of the multinomial response.

#### Variance Function for Multinomial

For observation $i$, the response vector $\mathbf{Y}_i = (Y_{i1}, \ldots, Y_{iJ})^\top$ follows $\text{Multinomial}(1, \boldsymbol{\pi}_i)$. The variance-covariance matrix is:

$$\text{Var}(\mathbf{Y}_i) = \boldsymbol{\Sigma}_i$$

with elements:

$$[\boldsymbol{\Sigma}_i]_{jk} = \text{Cov}(Y_{ij}, Y_{ik}) = \begin{cases} 
\pi_{ij}(1 - \pi_{ij}) & \text{if } j = k \\
-\pi_{ij}\pi_{ik} & \text{if } j \neq k
\end{cases}$$

This can be written compactly as:

$$\boldsymbol{\Sigma}_i = \text{diag}(\boldsymbol{\pi}_i) - \boldsymbol{\pi}_i \boldsymbol{\pi}_i^\top$$

or element-wise:

$$[\boldsymbol{\Sigma}_i]_{jk} = \pi_{ij}(\delta_{jk} - \pi_{ik})$$

#### Block Structure of Fisher Information

Since we only estimate $\boldsymbol{\beta}_1, \ldots, \boldsymbol{\beta}_{J-1}$ (with $\boldsymbol{\beta}_J = \mathbf{0}$ for identifiability), the Fisher information is a $(J-1)(p+1) \times (J-1)(p+1)$ block matrix:

$$\mathcal{I}(\boldsymbol{\beta}) = \begin{pmatrix}
\mathcal{I}_{11} & \mathcal{I}_{12} & \cdots & \mathcal{I}_{1,J-1} \\
\mathcal{I}_{21} & \mathcal{I}_{22} & \cdots & \mathcal{I}_{2,J-1} \\
\vdots & \vdots & \ddots & \vdots \\
\mathcal{I}_{J-1,1} & \mathcal{I}_{J-1,2} & \cdots & \mathcal{I}_{J-1,J-1}
\end{pmatrix}$$

where each $(p+1) \times (p+1)$ block is:

$$\mathcal{I}_{jk} = \sum_{i=1}^{n} \pi_{ij}(\delta_{jk} - \pi_{ik}) \mathbf{x}_i \mathbf{x}_i^\top$$

**Diagonal blocks** ($j = k$):
$$\mathcal{I}_{jj} = \sum_{i=1}^{n} \pi_{ij}(1 - \pi_{ij}) \mathbf{x}_i \mathbf{x}_i^\top = \mathbf{X}^\top \mathbf{W}_{jj} \mathbf{X}$$

where $\mathbf{W}_{jj} = \text{diag}(\pi_{1j}(1-\pi_{1j}), \ldots, \pi_{nj}(1-\pi_{nj}))$.

**Off-diagonal blocks** ($j \neq k$):
$$\mathcal{I}_{jk} = -\sum_{i=1}^{n} \pi_{ij}\pi_{ik} \mathbf{x}_i \mathbf{x}_i^\top = -\mathbf{X}^\top \mathbf{W}_{jk} \mathbf{X}$$

where $\mathbf{W}_{jk} = \text{diag}(\pi_{1j}\pi_{1k}, \ldots, \pi_{nj}\pi_{nk})$.

#### Asymptotic Variance of MLE

The maximum likelihood estimator $\hat{\boldsymbol{\beta}}$ is asymptotically normal:

$$\sqrt{n}(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}_0) \xrightarrow{d} \mathcal{N}\left(\mathbf{0}, \mathcal{I}^{-1}(\boldsymbol{\beta}_0)\right)$$

Standard errors are obtained from:

$$\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}) = \mathcal{I}^{-1}(\hat{\boldsymbol{\beta}}) = \left(\mathbf{X}^\top \hat{\mathbf{W}} \mathbf{X}\right)^{-1}$$

where $\hat{\mathbf{W}} = \mathbf{W}(\hat{\boldsymbol{\pi}})= \text{diag}(\hat{\pi}_{1j}(1-\hat{\pi}_{1j}), \ldots, \hat{\pi}_{nj}(1-\hat{\pi}_{nj}))$ is evaluated at $\hat{\boldsymbol{\pi}}$.
 
---

## Conditional Logistic Regression

### Motivation

In horse racing, we often have **matched data** where exactly one horse wins per race. Conditional logistic regression (McFadden's choice model) conditions on this constraint.

Further, matched data arises when we have multiple observations per race (e.g., per frame) but only one winner per race. This leads to a model that accounts for the within-race correlation by conditioning on the total number of winners (which is 1).

### Model Specification

For race $i$ with $J_i$ horses, let $Y_i \in \{1, \ldots, J_i\}$ indicate the winner. The conditional probability:

$$P(Y_i = j \mid \text{one horse wins}) = \frac{\exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta})}{\sum_{k=1}^{J_i} \exp(\mathbf{x}_{ik}^\top \boldsymbol{\beta})}$$

**Key difference from multinomial logit**: 
- Covariates $\mathbf{x}_{ij}$ vary by **both race $i$ and horse $j$**
- Only one coefficient vector $\boldsymbol{\beta}$ (not $J-1$ vectors)
- No intercept (absorbed by conditioning)

### Conditional Log-Likelihood

$$\ell_c(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ \mathbf{x}_{i,w_i}^\top \boldsymbol{\beta} - \log\left(\sum_{j=1}^{J_i} \exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta})\right) \right]$$

where $w_i$ is the index of the winning horse in race $i$.

### Equivalence to Cox Regression

The partial likelihood in Cox proportional hazards is mathematically identical to conditional logistic regression with matched sets corresponding to risk sets. Further, consider a survival dataset with $n$ individuals and $J_i$ events at time $t_i$. The risk set (set of individuals at risk) at time $t_i$ includes all individuals at risk just before $t_i$. The partial likelihood contribution for the event at $t_i$ is:

$$L_i(\boldsymbol{\beta}) = \frac{\exp(\mathbf{x}_{i,w_i}^\top \boldsymbol{\beta})}{\sum_{j=1}^{J_i} \exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta})} \iff \ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ \mathbf{x}_{i,w_i}^\top \boldsymbol{\beta} - \log\left(\sum_{j=1}^{J_i} \exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta})\right) \right]$$

This is exactly the same form as the conditional logistic regression likelihood.

---

## Estimation Theory

### Maximum Likelihood via IRLS

#### Binary Logistic Regression

The Newton-Raphson/Fisher Scoring update:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + \mathcal{I}^{-1}(\boldsymbol{\beta}^{(t)}) \cdot \mathbf{S}(\boldsymbol{\beta}^{(t)})$$

where the score is: $\mathbf{S}(\boldsymbol{\beta}) = \frac{\partial \ell}{\partial \boldsymbol{\beta}}$

**Iteratively Reweighted Least Squares (IRLS)** reformulation:

At iteration $t$:

1. Compute linear predictor: $\boldsymbol{\eta}^{(t)} = \mathbf{X}\boldsymbol{\beta}^{(t)}$

2. Compute fitted probabilities: $\boldsymbol{\pi}^{(t)} = \text{expit}(\boldsymbol{\eta}^{(t)})$

3. Compute weight matrix: $\mathbf{W}^{(t)} = \text{diag}\left(\pi_i^{(t)}(1 - \pi_i^{(t)})\right)$

4. Compute working response:
$$z_i^{(t)} = \eta_i^{(t)} + \frac{y_i - \pi_i^{(t)}}{\pi_i^{(t)}(1 - \pi_i^{(t)})}$$

1. Update coefficients via weighted least squares:
$$\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W}^{(t)} \mathbf{z}^{(t)}$$

1. Repeat until convergence: $\|\boldsymbol{\beta}^{(t+1)} - \boldsymbol{\beta}^{(t)}\|_\infty < \epsilon$

#### Working Response Derivation

The working response comes from a first-order Taylor expansion of $g(Y_i)$ around $\mu_i = \pi_i$:

$$g(Y_i) \approx g(\mu_i) + (Y_i - \mu_i) \cdot g'(\mu_i) = \eta_i + (y_i - \pi_i) \cdot \frac{1}{\pi_i(1-\pi_i)}$$

### Variance Estimation

The asymptotic covariance matrix of $\hat{\boldsymbol{\beta}}$:

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \mathcal{I}^{-1}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^\top \hat{\mathbf{W}} \mathbf{X})^{-1}$$

Standard errors: $\text{SE}(\hat{\beta}_j) = \sqrt{[\mathcal{I}^{-1}]_{jj}}$

### Multinomial IRLS

For multinomial logistic regression, the IRLS algorithm generalizes with:

- **Working response**: $\mathbf{z}_j^{(t)} = \boldsymbol{\eta}_j^{(t)} + \mathbf{W}_j^{-1}(\mathbf{y}_j - \boldsymbol{\pi}_j^{(t)})$
- **Weight matrix**: Block structure $\mathbf{W}_{jk} = \text{diag}(\pi_{ij}(\delta_{jk} - \pi_{ik}))$

The update requires solving a larger system with dimension $(J-1)(p+1)$.

---

## Model Diagnostics

### Deviance

The **deviance** measures the distance from the saturated model:

$$D = 2[\ell(\text{saturated}) - \ell(\text{fitted})]$$

#### Binary Logistic Regression

For individual binary observations:

$$D = 2 \sum_{i=1}^{n} \left[ y_i \log\frac{y_i}{\hat{\pi}_i} + (1-y_i) \log\frac{1-y_i}{1-\hat{\pi}_i} \right]$$

Using the convention $0 \log 0 = 0$:

$$D = -2 \sum_{i=1}^{n} \left[ y_i \log(\hat{\pi}_i) + (1-y_i) \log(1-\hat{\pi}_i) \right]$$

#### Multinomial Deviance

$$D = 2 \sum_{i=1}^{n} \sum_{j=1}^{J} y_{ij} \log\frac{y_{ij}}{\hat{\pi}_{ij}}$$

### Pearson Chi-Square Statistic

$$X^2 = \sum_{i=1}^{n} \frac{(y_i - \hat{\pi}_i)^2}{\hat{\pi}_i(1 - \hat{\pi}_i)}$$

For multinomial:

$$X^2 = \sum_{i=1}^{n} \sum_{j=1}^{J} \frac{(y_{ij} - \hat{\pi}_{ij})^2}{\hat{\pi}_{ij}}$$

### Residuals

#### Pearson Residuals

$$r_i^P = \frac{y_i - \hat{\pi}_i}{\sqrt{\hat{\pi}_i(1 - \hat{\pi}_i)}}$$

Properties: $\sum_i (r_i^P)^2 = X^2$

#### Deviance Residuals

$$r_i^D = \text{sign}(y_i - \hat{\pi}_i) \sqrt{d_i}$$

where $d_i$ is the contribution of observation $i$ to the deviance.

For binary:

$$d_i = 2\left[ y_i \log\frac{y_i}{\hat{\pi}_i} + (1-y_i) \log\frac{1-y_i}{1-\hat{\pi}_i} \right]$$

#### Standardized Residuals

Adjusting for leverage:

$$r_i^{PS} = \frac{y_i - \hat{\pi}_i}{\sqrt{\hat{\pi}_i(1-\hat{\pi}_i)(1-h_{ii})}}$$

where $h_{ii}$ is the $i$-th diagonal element of the hat matrix:

$$\mathbf{H} = \mathbf{W}^{1/2} \mathbf{X} (\mathbf{X}^\top \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W}^{1/2}$$

### Influence Diagnostics

#### Cook's Distance (Logistic Analog)

$$D_i = \frac{(r_i^{PS})^2 h_{ii}}{p(1 - h_{ii})}$$

#### DFBETAS

Change in coefficient estimates when observation $i$ is deleted:

$$\text{DFBETA}_{ij} = \frac{\hat{\beta}_j - \hat{\beta}_{j(i)}}{\text{SE}(\hat{\beta}_j)}$$

### Goodness-of-Fit Tests

#### Hosmer-Lemeshow Test

1. Sort observations by $\hat{\pi}_i$
2. Group into $G$ groups (typically 10)
3. Compute:

$$\hat{C} = \sum_{g=1}^{G} \frac{(O_g - E_g)^2}{E_g(1 - E_g/n_g)}$$

where $O_g = \sum_{i \in g} y_i$ and $E_g = \sum_{i \in g} \hat{\pi}_i$.

Under $H_0$ (adequate fit): $\hat{C} \stackrel{\cdot}{\sim} \chi^2_{G-2}$

---

## Implementation Considerations

### For Race Simulation

#### State Vector

Define the covariate vector for horse $j$ at time $t$:

$$\mathbf{x}_{jt} = \begin{pmatrix} 1 \\ d_{jt} \\ v_{jt} \\ a_{jt} \\ p_{jt} \end{pmatrix}$$

where:
- $d_{jt}$ = remaining distance to finish
- $v_{jt}$ = current velocity
- $a_{jt}$ = acceleration
- $p_{jt}$ = track position (inside/outside)

#### Cumulative Data Structure

Instead of fitting per-frame, accumulate observations:

$$\mathcal{D}_T = \{(\mathbf{x}_{jt}, y_{jt}) : j = 1, \ldots, J; \, t = 1, \ldots, T\}$$

where $y_{jt} = \mathbb{1}(\text{horse } j \text{ eventually wins})$.

#### Model Update Schedule

Fit the multinomial model every $\Delta t$ frames:

$$\hat{\boldsymbol{\beta}}_{t+\Delta t} = \underset{\boldsymbol{\beta}}{\arg\max} \, \ell(\boldsymbol{\beta}; \mathcal{D}_{t+\Delta t})$$

#### Real-Time Prediction

Given current state $\mathbf{x}_{j,t+1}$ and fitted $\hat{\boldsymbol{\beta}}_t$:

$$\hat{\pi}_{j,t+1} = \frac{\exp(\mathbf{x}_{j,t+1}^T \hat{\boldsymbol{\beta}}_j)}{\sum_{k=1}^{J} \exp(\mathbf{x}_{k,t+1}^T \hat{\boldsymbol{\beta}}_k)}$$

### Numerical Stability

#### Log-Sum-Exp Trick

To avoid overflow in expit:

$$\log\left(\sum_{j=1}^{J} e^{\eta_j}\right) = m + \log\left(\sum_{j=1}^{J} e^{\eta_j - m}\right)$$

where $m = \max_j \eta_j$.

#### Complete Separation

When a covariate perfectly predicts the outcome, MLE does not exist (infinite coefficients). Solutions:
- Firth's penalized likelihood
- Bayesian priors
- Data augmentation

### R Implementation (IRLS/Multinomial Log-Likelihood)

```r
# Core IRLS for binary logistic
irls.logistic <- function(X, y, tol = 1e-8, maxiter = 25) {
  beta <- rep(0, ncol(X))
  
  for (iter in seq_len(maxiter)) {
    eta <- X %*% beta
    pi <- plogis(eta)
    W <- diag(as.vector(pi * (1 - pi)))
    z <- eta + (y - pi) / (pi * (1 - pi))
    
    beta.new <- solve( t(X) %*% W %*% X, t(X) %*% W %*% z )
    
    if (max(abs(beta.new - beta)) < tol) break
    beta <- beta.new
  }
  
  list(
    coefficients = as.vector(beta),
    fitted = as.vector(plogis(X %*% beta)),
    vcov = solve(t(X) %*% W %*% X),
    iterations = iter
  )
}

# Multinomial logistic (expit/softmaxfunction)
expit <- function(eta.matrix) {
  eta.max <- apply(eta.matrix, 1, max)
  exp.eta <- exp(eta.matrix - eta.max)
  exp.eta / rowSums(exp.eta)
}

# Multinomial log-likelihood
multinom.log.lik <- function(beta.vec, X, Y, J) {
  p <- ncol(X)
  beta.mat <- matrix(c(beta.vec, rep(0, p)), nrow = p, ncol = J)
  eta <- X %*% beta.mat
  pi <- expit(eta)
  sum(Y * log(pi + 1e-10))
}
```


## References

1. Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.
2. McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
3. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
4. McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.
5. Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B (Methodological)*, 34(2), 187–202.**