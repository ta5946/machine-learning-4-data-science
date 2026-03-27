library(dplyr)
library(ggplot2)

datasets <- c("predictions/DummyClassifier_train_fold.csv",
              "predictions/LogisticRegression_nested_cv.csv",
              "predictions/KNeighborsClassifier_nested_cv.csv")

# Real world competition frequencies
real_weights <- c("NBA" = 0.6, "EURO" = 0.1, "SLO1" = 0.1, "U14" = 0.1, "U16" = 0.1)

for (dataset in datasets) {
  cat("Analyzing", dataset, "...\n")

  # Load dataframe
  df <- read.csv(dataset)
  
  # Create angle bins and binary error indicator
  bin_size <- 10
  df <- mutate(df,
               angle_bin = (Angle %/% bin_size) * bin_size + (bin_size / 2),
               error = as.numeric(y_true != y_pred))

  # Calculate mean error and standard deviation for each bin
  angle_analysis <- group_by(df, angle_bin)
  angle_analysis <- summarise(angle_analysis,
                              mean_error = mean(error),
                              se_error = sqrt((mean(error) * (1 - mean(error))) / n()),  # Binomial proportion
                              .groups = "drop")

  # Plot error rate by angle bin
  p <- ggplot(angle_analysis, aes(x = angle_bin, y = mean_error)) +  # Set x and y axis
    geom_col(fill = "steelblue", color = "white", width = bin_size) +
    geom_errorbar(aes(ymin = mean_error - se_error, ymax = mean_error + se_error),
                  width = 2, color = "black", alpha = 0.7) +
    scale_x_continuous(breaks = seq(0, 90, bin_size)) +
    ylim(0, 0.6) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    labs(title = paste("Error rate by angle for", dataset),
         x = "Angle bin",
         y = "Mean error")
  print(p)
  
  # Calculate and plot accuracy by competition
  comp_analysis <- group_by(df, Competition)
  comp_analysis <- summarise(comp_analysis,
                             accuracy = mean(y_true == y_pred),
                             .groups = "drop")
  print(comp_analysis)

  # Estimate real world accuracy by weighting accuracy per competition
  comp_analysis <- mutate(comp_analysis, weight = real_weights[Competition])
  est_accuracy <- sum(comp_analysis$accuracy * comp_analysis$weight)
  
  cat("Estimated real world accuracy:", round(est_accuracy, 4), "\n\n")
}
