# =============================================================================
# GLM/IRLS Engine for Horse Race Simulation
# Implements the exact specifications from GLM_TECHNICAL_SPECIFICATION.md
# =============================================================================

# -----------------------------------------------------------------------------
# BINARY LOGISTIC REGRESSION VIA IRLS
# -----------------------------------------------------------------------------

#' Compute the logistic (expit) function with numerical stability
#' @param eta Linear predictor values
#' @return Probabilities in (0, 1)
expit <- function(eta) {
  # Numerically stable sigmoid: avoid overflow for large |eta|
  ifelse(eta >= 0,
         1 / (1 + exp(-eta)),
         exp(eta) / (1 + exp(eta)))
}

#' Compute the logit (inverse of expit) function
#' @param p Probability values in (0, 1)
#' @return Linear predictor values
logit <- function(p) {
  log(p / (1 - p))
}

#' Binary Logistic Regression via Iteratively Reweighted Least Squares (IRLS)
#' 
#' Implements the Newton-Raphson/Fisher Scoring algorithm:
#'   β^(t+1) = β^(t) + I^(-1)(β^(t)) * U(β^(t))
#' 
#' Reformulated as weighted least squares:
#'   β^(t+1) = (X'W^(t)X)^(-1) X'W^(t)z^(t)
#' 
#' where:
#'   - W = diag(π_i(1 - π_i))  [variance function]
#'   - z_i = η_i + (y_i - π_i) / (π_i(1 - π_i))  [working response]
#' 
#' @param X Design matrix (n x p), should include intercept column
#' @param y Response vector (n x 1), binary 0/1
#' @param tol Convergence tolerance for coefficient change
#' @param maxiter Maximum number of IRLS iterations
#' @param verbose Print iteration information
#' @return List containing coefficients, fitted values, variance-covariance matrix, etc.
irls_logistic <- function(X, y, tol = 1e-8, maxiter = 25, verbose = FALSE) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize coefficients at zero
  beta <- rep(0, p)
  
  # IRLS iteration
  for (iter in seq_len(maxiter)) {
    # Step 1: Compute linear predictor
    eta <- as.vector(X %*% beta)
    
    # Step 2: Compute fitted probabilities via inverse link
    pi_hat <- expit(eta)
    
    # Step 3: Compute variance function V(μ) = π(1-π)
    # This is the diagonal of the weight matrix
    variance <- pi_hat * (1 - pi_hat)
    
    # Handle numerical issues: bound away from 0
    variance <- pmax(variance, 1e-10)
    
    # Step 4: Weight matrix W = diag(V(μ))
    W <- diag(variance)
    
    # Step 5: Working response (adjusted dependent variable)
    # z_i = η_i + (y_i - π_i) / V(μ_i)
    z <- eta + (y - pi_hat) / variance
    
    # Step 6: Weighted least squares update
    # β^(new) = (X'WX)^(-1) X'Wz
    XtWX <- t(X) %*% W %*% X
    XtWz <- t(X) %*% W %*% z
    
    # Solve the normal equations
    beta_new <- as.vector(solve(XtWX, XtWz))
    
    # Check convergence
    delta <- max(abs(beta_new - beta))
    if (verbose) {
      cat(sprintf("Iteration %d: max|Δβ| = %.2e\n", iter, delta))
    }
    
    if (delta < tol) {
      beta <- beta_new
      break
    }
    
    beta <- beta_new
  }
  
  # Final computations
  eta <- as.vector(X %*% beta)
  pi_hat <- expit(eta)
  variance <- pi_hat * (1 - pi_hat)
  variance <- pmax(variance, 1e-10)
  W <- diag(variance)
  
  # Fisher Information matrix I(β) = X'WX
  fisher_info <- t(X) %*% W %*% X
  
  # Variance-covariance matrix of β̂
  vcov <- solve(fisher_info)
  
  # Standard errors
  se <- sqrt(diag(vcov))
  
  # Deviance: D = -2 Σ [y_i log(π̂_i) + (1-y_i) log(1-π̂_i)]
  # Handle edge cases
  pi_safe <- pmax(pmin(pi_hat, 1 - 1e-10), 1e-10)
  deviance <- -2 * sum(y * log(pi_safe) + (1 - y) * log(1 - pi_safe))
  
  # Null deviance (intercept-only model)
  pi_null <- mean(y)
  null_deviance <- -2 * sum(y * log(pi_null) + (1 - y) * log(1 - pi_null))
  
  # Pearson residuals: r_i^P = (y_i - π̂_i) / sqrt(V(μ_i))
  pearson_resid <- (y - pi_hat) / sqrt(variance)
  
  # Deviance residuals
  deviance_contrib <- 2 * (y * log(pmax(y, 1e-10) / pi_safe) + 
                           (1 - y) * log(pmax(1 - y, 1e-10) / (1 - pi_safe)))
  deviance_resid <- sign(y - pi_hat) * sqrt(pmax(deviance_contrib, 0))
  
  # Pearson chi-square statistic
  pearson_chisq <- sum(pearson_resid^2)
  
  list(
    coefficients = beta,
    se = se,
    vcov = vcov,
    fitted = pi_hat,
    linear_predictor = eta,
    deviance = deviance,
    null_deviance = null_deviance,
    pearson_chisq = pearson_chisq,
    pearson_resid = pearson_resid,
    deviance_resid = deviance_resid,
    iterations = iter,
    converged = (iter < maxiter)
  )
}

# -----------------------------------------------------------------------------
# MULTINOMIAL LOGISTIC REGRESSION (SOFTMAX)
# -----------------------------------------------------------------------------

#' Softmax function with numerical stability (log-sum-exp trick)
#' 
#' For each row i:
#'   π_ij = exp(η_ij) / Σ_k exp(η_ik)
#' 
#' Using the log-sum-exp trick:
#'   log(Σ exp(η_j)) = m + log(Σ exp(η_j - m))  where m = max(η)
#' 
#' @param eta Matrix (n x J) of linear predictors
#' @return Matrix (n x J) of probabilities (rows sum to 1)
softmax <- function(eta) {
  # Subtract row max for numerical stability
  eta_max <- apply(eta, 1, max)
  eta_shifted <- eta - eta_max
  exp_eta <- exp(eta_shifted)
  exp_eta / rowSums(exp_eta)
}

#' Multinomial Log-Likelihood
#' 
#' ℓ(β) = Σ_i Σ_j y_ij [x_i'β_j - log(Σ_k exp(x_i'β_k))]
#' 
#' @param beta_vec Vectorized coefficients ((J-1)*p x 1)
#' @param X Design matrix (n x p)
#' @param Y Response matrix (n x J), one-hot encoded
#' @param J Number of categories
#' @return Negative log-likelihood (for minimization)
multinom_negloglik <- function(beta_vec, X, Y, J) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Reshape beta_vec to matrix: p x (J-1)
  # Reference category J has β_J = 0
  beta_mat <- matrix(beta_vec, nrow = p, ncol = J - 1)
  beta_mat <- cbind(beta_mat, rep(0, p))  # Add reference category
  
  # Linear predictors: n x J
  eta <- X %*% beta_mat
  
  # Probabilities via softmax
  pi_mat <- softmax(eta)
  
  # Log-likelihood
  # Avoid log(0)
  pi_safe <- pmax(pi_mat, 1e-10)
  loglik <- sum(Y * log(pi_safe))
  
  -loglik  # Return negative for minimization
}

#' Gradient of Multinomial Negative Log-Likelihood
#' 
#' ∂ℓ/∂β_j = X'(y_j - π_j)  for j = 1, ..., J-1
#' 
#' @param beta_vec Vectorized coefficients
#' @param X Design matrix
#' @param Y Response matrix (one-hot)
#' @param J Number of categories
#' @return Gradient vector
multinom_gradient <- function(beta_vec, X, Y, J) {
  n <- nrow(X)
  p <- ncol(X)
  
  beta_mat <- matrix(beta_vec, nrow = p, ncol = J - 1)
  beta_mat <- cbind(beta_mat, rep(0, p))
  
  eta <- X %*% beta_mat
  pi_mat <- softmax(eta)
  
  # Gradient: X'(Y - π) for each category except reference
  residuals <- Y - pi_mat
  
  # Stack gradients for j = 1, ..., J-1
  grad <- c()
  for (j in 1:(J - 1)) {
    grad <- c(grad, as.vector(t(X) %*% residuals[, j]))
  }
  
  -grad  # Negative for minimization
}

#' Multinomial Logistic Regression via Newton-Raphson
#' 
#' Uses optim() with BFGS and analytic gradients
#' 
#' @param X Design matrix (n x p), should include intercept
#' @param y Response vector (n x 1) with values in 1:J, or factor
#' @param maxiter Maximum iterations
#' @param verbose Print convergence info
#' @return List with coefficients, fitted probabilities, etc.
multinom_logistic <- function(X, y, maxiter = 100, verbose = FALSE) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Convert y to factor if needed
  if (!is.factor(y)) y <- as.factor(y)
  J <- nlevels(y)
  levels_y <- levels(y)
  
  # One-hot encode response
  Y <- matrix(0, nrow = n, ncol = J)
  for (i in 1:n) {
    Y[i, as.integer(y[i])] <- 1
  }
  
  # Initial coefficients (all zeros)
  beta_init <- rep(0, p * (J - 1))
  
  # Optimize using BFGS with analytic gradient
  result <- optim(
    par = beta_init,
    fn = multinom_negloglik,
    gr = multinom_gradient,
    X = X, Y = Y, J = J,
    method = "BFGS",
    control = list(maxit = maxiter, trace = verbose)
  )
  
  # Extract coefficients
  beta_mat <- matrix(result$par, nrow = p, ncol = J - 1)
  beta_mat <- cbind(beta_mat, rep(0, p))  # Add reference
  
  # Fitted probabilities
  eta <- X %*% beta_mat
  pi_mat <- softmax(eta)
  
  # Deviance
  pi_safe <- pmax(pi_mat, 1e-10)
  deviance <- -2 * sum(Y * log(pi_safe))
  
  # Null deviance (equal probabilities)
  pi_null <- colMeans(Y)
  null_deviance <- -2 * sum(Y * log(matrix(pi_null, nrow = n, ncol = J, byrow = TRUE)))
  
  list(
    coefficients = beta_mat,
    fitted = pi_mat,
    linear_predictor = eta,
    deviance = deviance,
    null_deviance = null_deviance,
    levels = levels_y,
    converged = (result$convergence == 0)
  )
}

# -----------------------------------------------------------------------------
# CONDITIONAL LOGISTIC REGRESSION (McFadden's Choice Model)
# -----------------------------------------------------------------------------

#' Conditional Logistic Regression
#' 
#' For matched data where exactly one alternative wins per choice set.
#' 
#' P(Y_i = j | one wins) = exp(x_ij'β) / Σ_k exp(x_ik'β)
#' 
#' ℓ_c(β) = Σ_i [x_{i,w_i}'β - log(Σ_j exp(x_ij'β))]
#' 
#' @param X_list List of design matrices, one per choice set
#' @param winner Vector of winner indices (which alternative won each set)
#' @param maxiter Maximum iterations
#' @return List with coefficients, etc.
conditional_logistic <- function(X_list, winner, maxiter = 100) {
  n_sets <- length(X_list)
  p <- ncol(X_list[[1]])
  
  # Negative log-likelihood
  negloglik <- function(beta) {
    ll <- 0
    for (i in seq_len(n_sets)) {
      X_i <- X_list[[i]]
      w_i <- winner[i]
      
      eta <- as.vector(X_i %*% beta)
      
      # Log-sum-exp trick
      eta_max <- max(eta)
      log_sum_exp <- eta_max + log(sum(exp(eta - eta_max)))
      
      ll <- ll + eta[w_i] - log_sum_exp
    }
    -ll
  }
  
  # Gradient
  gradient <- function(beta) {
    grad <- rep(0, p)
    for (i in seq_len(n_sets)) {
      X_i <- X_list[[i]]
      w_i <- winner[i]
      J_i <- nrow(X_i)
      
      eta <- as.vector(X_i %*% beta)
      pi_i <- softmax(matrix(eta, nrow = 1))[1, ]
      
      # Gradient contribution: x_{w_i} - Σ_j π_j x_j
      grad <- grad + X_i[w_i, ] - as.vector(t(X_i) %*% pi_i)
    }
    -grad
  }
  
  # Optimize
  beta_init <- rep(0, p)
  result <- optim(
    par = beta_init,
    fn = negloglik,
    gr = gradient,
    method = "BFGS",
    control = list(maxit = maxiter)
  )
  
  list(
    coefficients = result$par,
    loglik = -result$value,
    converged = (result$convergence == 0)
  )
}

# -----------------------------------------------------------------------------
# RACE SIMULATION ENGINE
# -----------------------------------------------------------------------------

#' Simulate Horse Race Dynamics
#' 
#' Generates race state evolution with physics-based movement
#' 
#' @param n_horses Number of horses
#' @param n_frames Number of simulation frames
#' @param race_length Total race distance (in radians for circular track)
#' @param base_speed Mean speed for all horses
#' @param speed_var Speed variation between horses
#' @param noise_sd Per-frame random noise in speed
#' @return List with positions, speeds, and winner
simulate_race <- function(n_horses = 10, 
                          n_frames = 500,
                          race_length = 3 * 2 * pi,  # 3 laps
                          base_speed = 0.02,
                          speed_var = 0.002,
                          noise_sd = 0.001) {
  
  # Initialize horse abilities (fixed per race)
  abilities <- rnorm(n_horses, mean = base_speed, sd = speed_var)
  abilities <- pmax(abilities, 0.005)  # Ensure positive
  
  # Initialize positions (all start at 0)
  positions <- matrix(0, nrow = n_frames, ncol = n_horses)
  speeds <- matrix(0, nrow = n_frames, ncol = n_horses)
  
  # Simulate frame by frame
  for (t in 2:n_frames) {
    # Current speed = ability + random noise
    current_speeds <- abilities + rnorm(n_horses, 0, noise_sd)
    current_speeds <- pmax(current_speeds, 0)
    
    speeds[t, ] <- current_speeds
    positions[t, ] <- positions[t - 1, ] + current_speeds
    
    # Check if race is over
    if (any(positions[t, ] >= race_length)) {
      # Truncate to finishing frame
      positions <- positions[1:t, , drop = FALSE]
      speeds <- speeds[1:t, , drop = FALSE]
      break
    }
  }
  
  # Determine winner
  final_positions <- positions[nrow(positions), ]
  winner <- which.max(final_positions)
  
  # Compute remaining distances at each frame
  remaining <- race_length - positions
  remaining[remaining < 0] <- 0
  
  list(
    positions = positions,
    speeds = speeds,
    remaining = remaining,
    winner = winner,
    n_horses = n_horses,
    n_frames = nrow(positions),
    race_length = race_length,
    abilities = abilities
  )
}

#' Build Design Matrix for GLM
#' 
#' Creates features from race state:
#'   - Intercept
#'   - Remaining distance (normalized)
#'   - Current speed (normalized)
#'   - Relative position (rank-based)
#' 
#' @param positions Current positions
#' @param speeds Current speeds
#' @param race_length Total race distance
#' @return Design matrix (n_horses x p)
build_design_matrix <- function(positions, speeds, race_length) {
  n <- length(positions)
  
  # Remaining distance (normalized to [0, 1])
  remaining <- (race_length - positions) / race_length
  
  # Speed (normalized)
  speed_norm <- (speeds - mean(speeds)) / (sd(speeds) + 1e-10)
  
  # Relative position (1 = leading, 0 = last)
  ranks <- rank(-positions)  # Higher position = lower rank
  rel_position <- 1 - (ranks - 1) / (n - 1)
  
  # Design matrix with intercept
  X <- cbind(
    intercept = 1,
    remaining = remaining,
    speed = speed_norm,
    rel_position = rel_position
  )
  
  X
}

#' Fit GLM and Predict Win Probabilities
#' 
#' Uses accumulated race data to fit multinomial logit and predict
#' current win probabilities
#' 
#' @param race_data Output from simulate_race()
#' @param current_frame Current frame index
#' @param history_frames How many past frames to use for fitting
#' @return Vector of win probabilities for each horse
predict_win_probabilities <- function(race_data, current_frame, history_frames = 50) {
  n_horses <- race_data$n_horses
  
  # Determine frames to use for fitting
  start_frame <- max(2, current_frame - history_frames + 1)
  frames <- start_frame:current_frame
  
  if (length(frames) < 10) {
    # Not enough data, return uniform probabilities
    return(rep(1 / n_horses, n_horses))
  }
  
  # Build cumulative dataset
  X_list <- list()
  y_vec <- c()
  
  for (t in frames) {
    X_t <- build_design_matrix(
      race_data$positions[t, ],
      race_data$speeds[t, ],
      race_data$race_length
    )
    X_list[[length(X_list) + 1]] <- X_t
    
    # Label: which horse eventually wins
    y_vec <- c(y_vec, race_data$winner)
  }
  
  # Stack into single dataset (for multinomial, each "race" is a time slice)
  # Here we use conditional logit since we have matched data
  
  # Fit conditional logistic
  tryCatch({
    fit <- conditional_logistic(X_list, rep(race_data$winner, length(frames)))
    
    # Predict probabilities at current frame
    X_current <- build_design_matrix(
      race_data$positions[current_frame, ],
      race_data$speeds[current_frame, ],
      race_data$race_length
    )
    
    eta <- as.vector(X_current %*% fit$coefficients)
    probs <- softmax(matrix(eta, nrow = 1))[1, ]
    
    probs
  }, error = function(e) {
    # Fallback: use distance-based heuristic
    remaining <- race_data$remaining[current_frame, ]
    inv_dist <- 1 / (remaining + 0.1)
    inv_dist / sum(inv_dist)
  })
}

# -----------------------------------------------------------------------------
# DIAGNOSTICS AND SUMMARIES
# -----------------------------------------------------------------------------

#' Print GLM Summary (similar to R's summary.glm)
#' 
#' @param fit Output from irls_logistic()
#' @param varnames Optional variable names
print_glm_summary <- function(fit, varnames = NULL) {
  p <- length(fit$coefficients)
  
  if (is.null(varnames)) {
    varnames <- paste0("X", 1:p)
  }
  
  # Z-statistics and p-values (Wald test)
  z_vals <- fit$coefficients / fit$se
  p_vals <- 2 * pnorm(-abs(z_vals))
  
  # Significance codes
  sig_codes <- ifelse(p_vals < 0.001, "***",
               ifelse(p_vals < 0.01, "**",
               ifelse(p_vals < 0.05, "*",
               ifelse(p_vals < 0.1, ".", " "))))
  
  cat("\nCoefficients:\n")
  coef_table <- data.frame(
    Estimate = round(fit$coefficients, 4),
    `Std. Error` = round(fit$se, 4),
    `z value` = round(z_vals, 3),
    `Pr(>|z|)` = format.pval(p_vals, digits = 3),
    ` ` = sig_codes
  )
  rownames(coef_table) <- varnames
  print(coef_table)
  
  cat("\n---\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n\n")
  
  cat(sprintf("Null deviance: %.3f\n", fit$null_deviance))
  cat(sprintf("Residual deviance: %.3f\n", fit$deviance))
  cat(sprintf("AIC: %.3f\n", fit$deviance + 2 * p))
  cat(sprintf("Number of IRLS iterations: %d\n", fit$iterations))
}

# -----------------------------------------------------------------------------
# VISUALIZATION HELPERS (for Shiny app)
# -----------------------------------------------------------------------------

#' Convert angular position to x,y coordinates on elliptical track
#' 
#' @param angle Angle in radians
#' @param a Semi-major axis (x-radius)
#' @param b Semi-minor axis (y-radius)
#' @return List with x and y coordinates
track_coords <- function(angle, a = 1.0, b = 0.5) {
  list(
    x = a * cos(angle),
    y = b * sin(angle)
  )
}

#' Generate color palette for horses
#' 
#' @param n_horses Number of horses
#' @return Vector of hex color codes
horse_colors <- function(n_horses) {
  rainbow(n_horses, alpha = 0.8)
}
