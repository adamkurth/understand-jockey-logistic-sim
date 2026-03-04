# =============================================================================
# Horse Race Simulation with GLM/IRLS Visualization
# Shiny Application
# =============================================================================

library(shiny)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# Source the GLM/IRLS engine
source("logistic.r")

# =============================================================================
# UI DEFINITION
# =============================================================================

ui <- fluidPage(
  # Custom CSS for styling
  tags$head(
    tags$style(HTML("
      body {
        background-color: #1a1a2e;
        color: #eaeaea;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      }
      .well {
        background-color: #16213e;
        border: 1px solid #0f3460;
      }
      .panel-heading {
        background-color: #0f3460 !important;
        color: #e94560 !important;
      }
      h1, h2, h3, h4 {
        color: #e94560;
      }
      .btn-primary {
        background-color: #e94560;
        border-color: #e94560;
      }
      .btn-primary:hover {
        background-color: #ff6b6b;
        border-color: #ff6b6b;
      }
      .probability-bar {
        height: 25px;
        margin: 2px 0;
        border-radius: 4px;
        transition: width 0.3s ease;
      }
      .math-display {
        background-color: #0f3460;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        overflow-x: auto;
        white-space: pre-wrap;
      }
      .leaderboard {
        background-color: #16213e;
        padding: 10px;
        border-radius: 8px;
      }
      .horse-row {
        display: flex;
        align-items: center;
        margin: 5px 0;
        padding: 5px;
        border-radius: 4px;
      }
      .horse-name {
        width: 80px;
        font-weight: bold;
      }
      .prob-value {
        width: 60px;
        text-align: right;
        font-family: monospace;
      }
    "))
  ),
  
  # Title
  titlePanel(
    div(
      h1("🏇 Horse Race Simulation", style = "margin-bottom: 0;"),
      h4("GLM/IRLS Real-Time Win Probability Estimation", 
         style = "color: #aaa; margin-top: 5px;")
    )
  ),
  
  # Layout
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      h4("Race Configuration"),
      
      sliderInput("n_horses", "Number of Horses:",
                  min = 3, max = 15, value = 8, step = 1),
      
      sliderInput("n_laps", "Number of Laps:",
                  min = 1, max = 5, value = 2, step = 1),
      
      sliderInput("competitiveness", "🔥 Competitiveness:",
                  min = 0, max = 1, value = 0.8, step = 0.1,
                  post = " (higher = more lead changes)"),
      
      sliderInput("speed_var", "Base Ability Spread:",
                  min = 0.001, max = 0.01, value = 0.004, step = 0.001),
      
      sliderInput("animation_speed", "Animation Speed (ms):",
                  min = 5, max = 100, value = 20, step = 5),
      
      hr(),
      
      actionButton("start_btn", "🏁 Start Race", 
                   class = "btn-primary btn-block"),
      
      actionButton("fast_btn", "⚡ Turbo Race (5ms)", 
                   class = "btn-warning btn-block"),
      
      actionButton("reset_btn", "🔄 Reset", 
                   class = "btn-secondary btn-block"),
      
      hr(),
      
      h4("GLM Settings"),
      
      selectInput("model_type", "Model Type:",
                  choices = c("Conditional Logit" = "conditional",
                              "Multinomial Logit" = "multinomial",
                              "Binary (Lead Horse)" = "binary")),
      
      sliderInput("update_interval", "Update Probability Every N Frames:",
                  min = 5, max = 50, value = 10, step = 5),
      
      checkboxInput("show_math", "Show IRLS Computations", value = TRUE),
      
      hr(),
      
      h4("About"),
      p("This simulation implements logistic regression via IRLS
         (Iteratively Reweighted Least Squares) following the
         GLM framework from Agresti (2013) and McCullagh & Nelder (1989).",
        style = "font-size: 11px; color: #888;")
    ),
    
    mainPanel(
      width = 9,
      
      fluidRow(
        # Race Track Visualization
        column(6,
          div(
            h3("Race Track"),
            plotOutput("race_plot", height = "350px"),
            style = "background-color: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 15px;"
          )
        ),
        
        # Win Probability Visualization
        column(6,
          div(
            h3("Win Probabilities (GLM Estimates)"),
            plotOutput("prob_plot", height = "350px"),
            style = "background-color: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 15px;"
          )
        )
      ),
      
      fluidRow(
        # Leaderboard
        column(4,
          div(
            h3("Current Standings"),
            uiOutput("leaderboard"),
            style = "background-color: #16213e; padding: 15px; border-radius: 8px;"
          )
        ),
        
        # Mathematical Details
        column(8,
          div(
            h3("IRLS Computations"),
            conditionalPanel(
              condition = "input.show_math",
              verbatimTextOutput("math_output", placeholder = TRUE)
            ),
            style = "background-color: #16213e; padding: 15px; border-radius: 8px;"
          )
        )
      ),
      
      # Progress indicator
      fluidRow(
        column(12,
          div(
            h4("Race Progress"),
            uiOutput("progress_bar"),
            textOutput("status_text"),
            style = "background-color: #16213e; padding: 15px; border-radius: 8px; margin-top: 15px;"
          )
        )
      )
    )
  )
)

# =============================================================================
# SERVER LOGIC
# =============================================================================

server <- function(input, output, session) {
  
  # Reactive values for race state
  rv <- reactiveValues(
    race_data = NULL,
    current_frame = 1,
    is_running = FALSE,
    probabilities = NULL,
    irls_info = "",
    colors = NULL
  )
  
  # Initialize/reset race
  observeEvent(input$reset_btn, {
    rv$race_data <- NULL
    rv$current_frame <- 1
    rv$is_running <- FALSE
    rv$probabilities <- NULL
    rv$irls_info <- ""
  })
  
  # Helper function to start a race
  start_new_race <- function(turbo = FALSE) {
    # Generate new race with competitive dynamics
    rv$race_data <- simulate_race(
      n_horses = input$n_horses,
      n_frames = 1500,
      race_length = input$n_laps * 2 * pi,
      speed_var = input$speed_var,
      noise_sd = 0.003,
      competitiveness = input$competitiveness
    )
    
    # Generate colors
    rv$colors <- rainbow(input$n_horses, alpha = 0.9)
    
    # Initialize probabilities (uniform)
    rv$probabilities <- rep(1 / input$n_horses, input$n_horses)
    
    rv$current_frame <- 1
    rv$is_running <- TRUE
    
    if (turbo) {
      # Update animation speed to minimum
      updateSliderInput(session, "animation_speed", value = 5)
    }
  }
  
  # Start race (normal)
  observeEvent(input$start_btn, {
    if (!rv$is_running) {
      start_new_race(turbo = FALSE)
    }
  })
  
  # Start race (turbo mode)
  observeEvent(input$fast_btn, {
    if (!rv$is_running) {
      start_new_race(turbo = TRUE)
    }
  })
  
  # Animation timer
  observe({
    if (rv$is_running && !is.null(rv$race_data)) {
      invalidateLater(input$animation_speed, session)
      
      isolate({
        if (rv$current_frame < rv$race_data$n_frames) {
          rv$current_frame <- rv$current_frame + 1
          
          # Update probabilities at intervals
          if (rv$current_frame %% input$update_interval == 0) {
            update_probabilities()
          }
        } else {
          rv$is_running <- FALSE
        }
      })
    }
  })
  
  # Update probabilities using GLM
  update_probabilities <- function() {
    if (is.null(rv$race_data) || rv$current_frame < 20) return()
    
    n_horses <- rv$race_data$n_horses
    t <- rv$current_frame
    
    # Build design matrix for current state
    X_current <- build_design_matrix(
      rv$race_data$positions[t, ],
      rv$race_data$speeds[t, ],
      rv$race_data$race_length
    )
    
    if (input$model_type == "binary") {
      # Binary logistic: model probability of being in the lead
      # Use historical data where we know which horse led
      
      # Collect data from past frames
      start_t <- max(2, t - 100)
      frames <- start_t:t
      
      X_all <- c()
      y_all <- c()
      
      for (s in frames) {
        X_s <- build_design_matrix(
          rv$race_data$positions[s, ],
          rv$race_data$speeds[s, ],
          rv$race_data$race_length
        )
        
        # Leader at frame s
        leader <- which.max(rv$race_data$positions[s, ])
        y_s <- rep(0, n_horses)
        y_s[leader] <- 1
        
        X_all <- rbind(X_all, X_s)
        y_all <- c(y_all, y_s)
      }
      
      # Fit binary logistic via Ridge-IRLS (lambda prevents separation issues)
      tryCatch({
        fit <- irls_logistic(X_all, y_all, lambda = 0.5, verbose = FALSE)
        
        # Predict probabilities for current state (clip eta for stability)
        eta_current <- as.vector(X_current %*% fit$coefficients)
        eta_current <- pmax(pmin(eta_current, 10), -10)
        probs <- expit(eta_current)
        probs <- probs / sum(probs)  # Normalize to sum to 1
        
        rv$probabilities <- probs
        
        # Count lead changes for drama indicator
        leaders <- apply(rv$race_data$positions[1:t, , drop=FALSE], 1, which.max)
        lead_changes <- sum(diff(leaders) != 0)
        
        # Update IRLS info
        rv$irls_info <- sprintf(
"RIDGE-REGULARIZED BINARY LOGISTIC REGRESSION (IRLS)
====================================================
Frame: %d | Observations: %d | λ (ridge): 0.5

RACE DYNAMICS:
  Lead Changes: %d  %s

COEFFICIENTS (with L2 penalty):
  Intercept     : %+.4f (SE: %.4f)
  Remaining Dist: %+.4f (SE: %.4f)  
  Speed         : %+.4f (SE: %.4f)
  Rel. Position : %+.4f (SE: %.4f)

MODEL FIT:
  Null Deviance    : %.2f
  Residual Deviance: %.2f
  IRLS Iterations  : %d
  Converged        : %s

CURRENT LINEAR PREDICTORS (η = Xβ, clipped to [-10, 10]):
%s

WIN PROBABILITIES (softmax normalized):
%s",
          t, nrow(X_all), lead_changes,
          ifelse(lead_changes > 5, "🔥 TIGHT RACE!", 
                 ifelse(lead_changes > 2, "⚡ Competitive", "📊 Steady leader")),
          fit$coefficients[1], fit$se[1],
          fit$coefficients[2], fit$se[2],
          fit$coefficients[3], fit$se[3],
          fit$coefficients[4], fit$se[4],
          fit$null_deviance, fit$deviance, fit$iterations,
          ifelse(fit$converged, "YES", "NO"),
          paste(sprintf("  Horse %2d: η = %+6.2f", 1:n_horses, eta_current), collapse = "\n"),
          paste(sprintf("  Horse %2d: %5.1f%% %s", 1:n_horses, probs * 100,
                        ifelse(probs == max(probs), "← FAVORITE", "")), collapse = "\n")
        )
        
      }, error = function(e) {
        # Fallback to distance-based
        remaining <- rv$race_data$remaining[t, ]
        probs <- 1 / (remaining + 0.1)
        rv$probabilities <- probs / sum(probs)
        rv$irls_info <- paste("IRLS failed:", e$message, "\nUsing distance-based heuristic.")
      })
      
    } else if (input$model_type == "conditional") {
      # Conditional logit (McFadden's choice model)
      
      start_t <- max(2, t - 50)
      frames <- start_t:t
      
      X_list <- list()
      for (s in frames) {
        X_s <- build_design_matrix(
          rv$race_data$positions[s, ],
          rv$race_data$speeds[s, ],
          rv$race_data$race_length
        )
        X_list[[length(X_list) + 1]] <- X_s
      }
      
      # Winner is known (for this demo, we "peek" at final result)
      winners <- rep(rv$race_data$winner, length(frames))
      
      tryCatch({
        fit <- conditional_logistic(X_list, winners)
        
        # Predict probabilities
        eta_current <- as.vector(X_current %*% fit$coefficients)
        probs <- softmax(matrix(eta_current, nrow = 1))[1, ]
        
        rv$probabilities <- probs
        
        rv$irls_info <- sprintf(
"CONDITIONAL LOGISTIC REGRESSION
================================
Frame: %d | Choice Sets: %d

COEFFICIENTS (no intercept - absorbed by conditioning):
  Remaining Dist: %+.4f
  Speed         : %+.4f
  Rel. Position : %+.4f

Log-Likelihood: %.2f
Converged: %s

CURRENT LINEAR PREDICTORS:
%s

WIN PROBABILITIES (softmax):
%s",
          t, length(X_list),
          fit$coefficients[2], fit$coefficients[3], fit$coefficients[4],
          fit$loglik, ifelse(fit$converged, "YES", "NO"),
          paste(sprintf("  Horse %2d: η = %+.3f", 1:n_horses, eta_current), collapse = "\n"),
          paste(sprintf("  Horse %2d: π = %.1f%%", 1:n_horses, probs * 100), collapse = "\n")
        )
        
      }, error = function(e) {
        remaining <- rv$race_data$remaining[t, ]
        probs <- 1 / (remaining + 0.1)
        rv$probabilities <- probs / sum(probs)
        rv$irls_info <- paste("Conditional logit failed:", e$message)
      })
      
    } else {
      # Multinomial logit
      # This is computationally heavier, use sparingly
      
      start_t <- max(2, t - 30)
      frames <- start_t:t
      
      X_all <- c()
      y_all <- c()
      
      for (s in frames) {
        # Use leader at each frame as "winner" for that slice
        leader <- which.max(rv$race_data$positions[s, ])
        y_all <- c(y_all, leader)
        
        # Use average features across horses for that frame
        X_s <- build_design_matrix(
          rv$race_data$positions[s, ],
          rv$race_data$speeds[s, ],
          rv$race_data$race_length
        )
        X_all <- rbind(X_all, colMeans(X_s))
      }
      
      # Fallback: use relative positions as probabilities
      remaining <- rv$race_data$remaining[t, ]
      probs <- 1 / (remaining + 0.1)
      rv$probabilities <- probs / sum(probs)
      
      rv$irls_info <- sprintf(
"MULTINOMIAL LOGIT (distance-based approximation)
=================================================
Frame: %d

Using softmax on inverse remaining distances:
  π_j ∝ 1 / (d_j + ε)

%s",
        t,
        paste(sprintf("  Horse %2d: d = %.3f, π = %.1f%%", 
                      1:n_horses, remaining, probs * 100), collapse = "\n")
      )
    }
  }
  
  # Race track plot
  output$race_plot <- renderPlot({
    if (is.null(rv$race_data)) {
      # Empty track
      ggplot() +
        annotate("path",
                 x = cos(seq(0, 2*pi, length.out = 100)),
                 y = 0.5 * sin(seq(0, 2*pi, length.out = 100)),
                 color = "#444", size = 2) +
        annotate("text", x = 0, y = 0, label = "Press 'Start Race' to begin",
                 color = "#888", size = 6) +
        coord_fixed(xlim = c(-1.3, 1.3), ylim = c(-0.8, 0.8)) +
        theme_void() +
        theme(
          plot.background = element_rect(fill = "#16213e", color = NA),
          panel.background = element_rect(fill = "#16213e", color = NA)
        )
    } else {
      t <- rv$current_frame
      n_horses <- rv$race_data$n_horses
      
      # Get current positions (modulo 2π for track position)
      angles <- rv$race_data$positions[t, ] %% (2 * pi)
      
      # Convert to x,y coordinates
      horse_x <- cos(angles)
      horse_y <- 0.5 * sin(angles)
      
      # Create data frame
      horse_df <- data.frame(
        horse = factor(1:n_horses),
        x = horse_x,
        y = horse_y,
        color = rv$colors
      )
      
      # Track outline
      track_angles <- seq(0, 2*pi, length.out = 100)
      track_df <- data.frame(
        x = cos(track_angles),
        y = 0.5 * sin(track_angles)
      )
      
      # Finish line
      finish_df <- data.frame(
        x = c(1, 1),
        y = c(-0.1, 0.1)
      )
      
      ggplot() +
        # Track
        geom_path(data = track_df, aes(x = x, y = y),
                  color = "#444", size = 8, lineend = "round") +
        geom_path(data = track_df, aes(x = x, y = y),
                  color = "#2d4a22", size = 6, lineend = "round") +
        # Lane markings
        geom_path(data = track_df, aes(x = x * 0.85, y = y * 0.85),
                  color = "#555", size = 0.5, linetype = "dashed") +
        geom_path(data = track_df, aes(x = x * 1.15, y = y * 1.15),
                  color = "#555", size = 0.5, linetype = "dashed") +
        # Finish line
        geom_segment(aes(x = 1, xend = 1, y = -0.15, yend = 0.15),
                     color = "white", size = 3) +
        geom_segment(aes(x = 1, xend = 1, y = -0.15, yend = 0.15),
                     color = "black", size = 2, linetype = "dashed") +
        # Horses
        geom_point(data = horse_df, aes(x = x, y = y, color = horse),
                   size = 6) +
        geom_text(data = horse_df, aes(x = x, y = y, label = horse),
                  color = "white", size = 3, fontface = "bold") +
        scale_color_manual(values = rv$colors) +
        coord_fixed(xlim = c(-1.4, 1.4), ylim = c(-0.9, 0.9)) +
        theme_void() +
        theme(
          plot.background = element_rect(fill = "#16213e", color = NA),
          panel.background = element_rect(fill = "#16213e", color = NA),
          legend.position = "none"
        )
    }
  }, bg = "#16213e")
  
  # Probability bar chart
  output$prob_plot <- renderPlot({
    if (is.null(rv$probabilities) || is.null(rv$race_data)) {
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "Waiting for race data...",
                 color = "#888", size = 5) +
        theme_void() +
        theme(
          plot.background = element_rect(fill = "#16213e", color = NA),
          panel.background = element_rect(fill = "#16213e", color = NA)
        )
    } else {
      n_horses <- rv$race_data$n_horses
      
      prob_df <- data.frame(
        horse = factor(1:n_horses),
        probability = rv$probabilities,
        color = rv$colors
      )
      
      # Sort by probability
      prob_df <- prob_df[order(-prob_df$probability), ]
      prob_df$horse <- factor(prob_df$horse, levels = prob_df$horse)
      
      ggplot(prob_df, aes(x = horse, y = probability, fill = horse)) +
        geom_col(width = 0.7) +
        geom_text(aes(label = sprintf("%.1f%%", probability * 100)),
                  vjust = -0.5, color = "white", size = 3.5) +
        scale_fill_manual(values = setNames(rv$colors, 1:n_horses)) +
        scale_y_continuous(labels = percent_format(), limits = c(0, max(rv$probabilities) * 1.2)) +
        labs(x = "Horse", y = "Win Probability") +
        theme_minimal() +
        theme(
          plot.background = element_rect(fill = "#16213e", color = NA),
          panel.background = element_rect(fill = "#16213e", color = NA),
          panel.grid = element_line(color = "#333"),
          axis.text = element_text(color = "#eaeaea"),
          axis.title = element_text(color = "#eaeaea"),
          legend.position = "none"
        )
    }
  }, bg = "#16213e")
  
  # Leaderboard
  output$leaderboard <- renderUI({
    if (is.null(rv$race_data)) {
      return(div("No race in progress", style = "color: #888;"))
    }
    
    t <- rv$current_frame
    n_horses <- rv$race_data$n_horses
    
    # Current positions
    positions <- rv$race_data$positions[t, ]
    ranks <- rank(-positions)
    
    # Create leaderboard entries
    entries <- lapply(order(ranks), function(i) {
      div(
        class = "horse-row",
        style = sprintf("background-color: %s33;", rv$colors[i]),
        span(class = "horse-name", 
             style = sprintf("color: %s;", rv$colors[i]),
             sprintf("Horse %d", i)),
        span(sprintf("Rank: %d", ranks[i]), 
             style = "flex-grow: 1; text-align: center;"),
        span(class = "prob-value",
             sprintf("%.1f%%", rv$probabilities[i] * 100))
      )
    })
    
    do.call(div, c(list(class = "leaderboard"), entries))
  })
  
  # IRLS math output
  output$math_output <- renderText({
    if (rv$irls_info == "") {
      return("IRLS computations will appear here once the race starts and enough data is collected...")
    }
    rv$irls_info
  })
  
  # Progress bar
  output$progress_bar <- renderUI({
    if (is.null(rv$race_data)) {
      return(div(
        style = "height: 20px; background-color: #333; border-radius: 10px;",
        div(style = "width: 0%; height: 100%; background-color: #e94560; border-radius: 10px;")
      ))
    }
    
    progress <- rv$current_frame / rv$race_data$n_frames * 100
    
    div(
      style = "height: 20px; background-color: #333; border-radius: 10px;",
      div(
        style = sprintf("width: %.1f%%; height: 100%%; background-color: #e94560; border-radius: 10px; transition: width 0.1s;", progress)
      )
    )
  })
  
  # Status text
  output$status_text <- renderText({
    if (is.null(rv$race_data)) {
      return("Ready to start")
    }
    
    t <- rv$current_frame
    total <- rv$race_data$n_frames
    
    if (t >= total) {
      winner <- rv$race_data$winner
      return(sprintf("🏆 Race Complete! Winner: Horse %d", winner))
    }
    
    sprintf("Frame %d / %d (%.1f%% complete)", t, total, t/total * 100)
  })
}

# =============================================================================
# RUN APPLICATION
# =============================================================================

shinyApp(ui = ui, server = server)
