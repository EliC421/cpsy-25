#!/usr/bin/env rscript

#usage:
# cd /path/to/code/func/
# chmod +x accel.r
# ./accel.r --project_dir="path/to/project" --project_deriv_dir="path/to/derivatives" --files="file1.csv, file2.csv, file3.csv" --verbose

# load required libraries
library(optparse)
library(tidyr)
library(plyr)
library(GGIR)

# define command-line options
option_list <- list(
  make_option(c("-p", "--project_dir"), type="character", default="\\\\itf-rs-store24.hpc.uiowa.edu\\vosslabhpc\\symposia\\cpsy-25\\temp",
              help="project directory [default= %default]", metavar="character"),
  make_option(c("-d", "--project_deriv_dir"), type="character", default="C:\\Users\\jedim\\Voss_Lab\\cpsy-25\\pacrad_ggiroutput",
              help="project derivative directory [default= %default]", metavar="character"),
  make_option(c("-f", "--files"), type="character", default=NULL,
              help="comma-separated list of files to process", metavar="character"),
  make_option(c("-v", "--verbose"), action="store_true", default=TRUE,
              help="print verbose output [default= %default]")
)

# parse command-line arguments
opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)
print(opt)
cat("project directory:", opt$project_dir, "\n")
cat("files:", opt$files, "\n")

# Read the completed subjects text file
completed_subjects <- readLines("C:/Users/jedim/Voss_Lab/cpsy-25/completed_subjects.txt")
print("Completed subjects file read successfully")

# Parse the completed subjects into a data frame
completed_labids <- data.frame(
  Lab.ID = as.numeric(sub("Lab ID: (\\d+),.*", "\\1", completed_subjects)),
  PACR.Study.ID = as.numeric(sub(".*Study ID: (\\d+).*", "\\1", completed_subjects))
)
print("Parsed completed subjects")

# Read the subject paths text file
subject_paths <- readLines("C:/Users/jedim/Voss_Lab/cpsy-25/subject_paths.txt")
print("Subject paths file read successfully")

# Extract Lab IDs from subject paths
subject_paths_df <- data.frame(
  Path = subject_paths,
  Lab.ID = as.numeric(sub(".*\\\\(\\d+) \\(.*\\)RAW.*", "\\1", subject_paths))
)
print("Parsed subject paths")

# main function
main <- function(opt) {
  # set paths
  paths <- set_paths(opt$project_dir, opt$project_deriv_dir)
  print("Paths set successfully")
  
  # Skip the first file and process the rest
  for (i in 2:nrow(subject_paths_df)) {
    raw_data_file <- subject_paths_df$Path[i]
    lab_id <- subject_paths_df$Lab.ID[i]
    
    # Find the corresponding Study ID
    pacr_study_id <- completed_labids[completed_labids$Lab.ID == lab_id, "PACR.Study.ID"]
    if (length(pacr_study_id) == 0) {
      print(paste("No matching Study ID found for Lab ID:", lab_id))
      next
    }
    pacr_study_id <- pacr_study_id[1]
    print(paste("Processing Lab ID:", lab_id, "PACR Study ID:", pacr_study_id))
    
    # Check if the raw data file exists
    if (!file.exists(raw_data_file)) {
      print(paste("Raw data file does not exist:", raw_data_file))
      next
    }
    
    # Process the raw data file
    print(paste("Processing raw file:", raw_data_file))
    process_path(raw_data_file, paths$projectderivdir, pacr_study_id, opt$verbose)
  }
}

# function to set paths
set_paths <- function(project_dir, project_deriv_dir) {
  projectdir <- normalizePath(project_dir, mustWork = FALSE)
  projectderivdir <- normalizePath(project_deriv_dir, mustWork = FALSE)
  return(list(projectdir=projectdir, projectderivdir=projectderivdir))
}

# Function to process each path and create part 2-5 summaries
process_path <- function(path, base_output_dir, pacr_study_id, verbose) {
  # Set dir_path to the directory containing the raw data file
  dir_path <- dirname(path)
  
  # Print the dir_path for debugging
  if (verbose) {
    print(paste("dir_path:", dir_path))
  }
  
  # Set the output directory for the summaries
  output_dir <- file.path(base_output_dir, paste0("sub-", pacr_study_id), "ses-pre", "beh", "output_beh", "results")
  
  # Print the output_dir for debugging
  if (verbose) {
    print(paste("output_dir:", output_dir))
  }
  
  # Check if the directory exists and is writable
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    if (verbose) {
      print(paste("Created directory:", output_dir))
    }
  } else if (!file.access(output_dir, 2) == 0) {
    stop(paste("User does not have write access permissions for the directory:", output_dir))
  } else {
    if (verbose) {
      print(paste("Directory exists and is writable:", output_dir))
    }
  }
  
  # Print the variables before calling GGIR for debugging
  if (verbose) {
    print(paste("Calling GGIR with dir_path:", dir_path, "and output_dir:", output_dir))
  }
  
  # Run GGIR to create part 2-5 summaries
  GGIR::GGIR(
    mode = 1:5,
    datadir = dir_path,
    outputdir = output_dir,
    studyname = "boost",
    overwrite = TRUE,
    print.filename = TRUE,
    storefolderstructure = FALSE,
    windowsizes = c(5, 900, 3600),
    desiredtz = "america/chicago",
    do.enmo = TRUE,   
    do.anglez = TRUE,
    # … additional parameters as needed
  )
  
  print(paste("Processed:", path))
}

# run the main function
main(opt)

