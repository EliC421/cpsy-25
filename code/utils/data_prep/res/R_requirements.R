# Set CRAN repository
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Install required packages
install.packages("GGIR", dependencies = TRUE)
install.packages("optparse")
install.packages("tidyr")
install.packages("plyr")

# Load required libraries
library(GGIR)
library(optparse)
library(tidyr)
library(plyr)
