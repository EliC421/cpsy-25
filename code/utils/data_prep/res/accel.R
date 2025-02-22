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
  make_option(c("-p", "--project_dir"), type="character", default="\\\\itf-rs-store24.hpc.uiowa.edu\\vosslabhpc\\Projects\\BETTER\\3-Experiment\\2-data\\bids",
              help="project directory [default= %default]", metavar="character"),
  make_option(c("-d", "--project_deriv_dir"), type="character", default="derivatives/GGIR_8.2.8/testing",
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
# main function
main <- function(opt) {
  # set paths
  paths <- set_paths(opt$project_dir, opt$project_deriv_dir)
  ggirfiles <- character(0)

  # check if files were provided as input
  if (!is.NULL(opt$files)) {
    # split the files provided as a comma-separated string
    files_to_process <- unlist(strsplit(opt$files, split=","))
    
    # ensure the provided files exist
    ggirfiles <- files_to_process[file.exists(files_to_process)]
    
    # ensure ggirfiles is a character vector and not empty
    if (is.NULL(ggirfiles) || length(ggirfiles) == 0) {
      stop("ggirfiles is not initialized or empty or the specified files do not exist.")
    }

    cat("ggirfiles before processing:", ggirfiles, "\n")
    
    # process each file directly
    for (r in ggirfiles) {
      process_file(r, paths$projectdir, paths$projectderivdir, opt$verbose)
    }
    
    return()  # exit the function after processing the files
  } 

  # if no files are provided, proceed with the preprocessing steps
  # gather subject directories
  subdirs <- gather_subject_directories(paths$projectdir)

  # create output directories
  create_output_directories(paths$projectdir, paths$projectderivdir)

  # list accel.csv files from subject directories
  ggirfiles <- list_accel_files(subdirs, paths$projectdir)

  # ensure ggirfiles is a character vector and not empty
  if (is.NULL(ggirfiles) || length(ggirfiles) == 0) {
    stop("ggirfiles is not initialized or empty.")
  }

  if (!is.character(ggirfiles)) {
    stop("ggirfiles must be a character vector.")
  }

  # process each file found in the directories
  for (r in ggirfiles) {
    process_file(r, paths$projectdir, paths$projectderivdir, opt$verbose)
  }
}

# function to set paths
set_paths <- function(project_dir, project_deriv_dir) {
  projectdir <- normalizepath(project_dir, mustwork = false)
  projectderivdir <- project_deriv_dir
  return(list(projectdir=projectdir, projectderivdir=projectderivdir))
}

# function to gather subject directories
gather_subject_directories <- function(project_dir) {
  directories <- list.dirs(project_dir, recursive = false)
  subdirs <- directories[grepl("[0-9]", directories)]
  return(subdirs)
}

# function to create output directories
create_output_directories <- function(project_dir, project_deriv_dir) {
  output_dir <- file.path(project_dir, project_deriv_dir)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = true)
  }
}

# function to list accel.csv files
list_accel_files <- function(subdirs, project_dir) {
  filepattern <- "*.csv"
  ggirfiles <- list()
  for (subdir in subdirs) {
    files_in_subdir <- list.files(subdir, pattern=filepattern, recursive=true,
                                  include.dirs=true, full.names=true, no..=true)
    ggirfiles <- c(ggirfiles, files_in_subdir)
  }
  # get relative paths
  ggirfiles <- sapply(ggirfiles, function(x) {
    rel_path <- substr(x, nchar(project_dir)+1, nchar(x))
    return(rel_path)
  })
  return(ggirfiles)
}

# function to process each file
process_file <- function(r, project_dir, project_deriv_dir, verbose=FALSE) {
  # Helper functions
  subjectggirderiv <- function(x) {
    output <- file.path(project_dir, project_deriv_dir)
    return(output)
  }

  datadirname <- function(x) {
    b <- dirname(x)
    outputname <- file.path(b)
    return(outputname)
  }

  # Define directories
  outputdir <- subjectggirderiv(r)
  datadir <- datadirname(r)

  # Create directory if it doesn't exist
  if (!dir.exists(outputdir)) {
    dir.create(outputdir, recursive = TRUE)
  }

  # Skip if output already exists
  if (dir.exists(file.path(outputdir, "output_beh"))) {
    if (verbose) cat("skipping", r, "- output already exists.\n")
    return()
  }
  if (!dir.exists(datadir)) {
    stop("data directory does not exist: ", datadir)
  } else {
    print("data directory exists")
  }
  # Run GGIR
  tryCatch({
    if (verbose) {
      cat("processing file:", r, "\n")
      cat("data directory:", datadir, "\n")
      cat("output directory:", outputdir, "\n")
    }
    datadir <<- datadir  # assign as a global variable
    outputdir <<- outputdir  # assign as a global variable
    print(r)
    session <<- strsplit("/", r)[[-2]]
    #print session
    print(session)
    GGIR::GGIR(
      mode = 1:6,
      datadir = datadir,
      outputdir = outputdir,
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
    if (verbose) cat("ggir processing completed for", r, "\n")
    
    # Post-processing
    post_process_file(r, project_dir, project_deriv_dir, verbose)
    
    # Re-run ggir part 5
    #re_run_ggir_part5(r, project_dir, project_deriv_dir, verbose)
    
    # Re-run intensity gradient calculations
    #re_run_intensity_gradient(r, project_dir, project_deriv_dir, verbose)
    
  }, error=function(e) {
    cat("error processing", r, ":", e$message, "\n")
  })
}

# function for post-processing
post_process_file <- function(r, project_dir, project_deriv_dir, verbose=false) {
  subjectggirderiv <- function(x) {
    ses <- strsplit("/", x)[[-2]]
    output <- file.path(project_dir, project_deriv_dir, ses)
    return(output)
  }
  
  session <- strsplit("/", r)[[-2]]
  outputdir <- subjectggirderiv(r)
  output_ms5 <- file.path(paste0(outputdir, "output_", session, "/meta/ms5.out"))
  output_ms5 <- file.path(paste0(outputdir, "output_", session, "/meta/ms5.out_original"))
  
  if (dir.exists(output_ms5_original)) {
    if (verbose) cat("post-processing already done for", r, "\n")
    return()
  }
  
  # rename ms5.out to ms5.out_original
  if (dir.exists(output_ms5)) {
    file.rename(output_ms5, output_ms5_original)
  } else {
    if (verbose) cat("ms5.out not found for", r, "\n")
    return()
  }
  
  # load, clean, and save data
  dir <- output_ms5_original
  files <- list.files(dir)
  if (length(files) == 0) {
    if (verbose) cat("no files to process in", dir, "\n")
    return()
  }
  
  if (verbose) cat("post-processing", length(files), "files for", r, "\n")
  
  # initialize data frames
  removed <- data.frame(crit = c("1","3","4b","totals"), 
                        nights = rep(0,4), 
                        participants_affected = rep(0,4),
                        participants_no_valid = rep(0,4), 
                        participants_zero_days = rep(0,4))
  removed_person <- data.frame(id = files, 
                               nights_crit1 = 0, 
                               nights_crit3 = 0,
                               nights_crit4b = 0,
                               nights_allcrit = 0)
  datacleanmm <- data.frame()
  datacleanww <- data.frame()
  
  for (i in seq_along(files)) {
    load(file.path(dir, files[i]))
    # [insert data cleaning code here]
    # save cleaned data
    output_ms5_clean <- file.path(outputdir, "output_beh/meta/ms5.out/")
    if (!dir.exists(output_ms5_clean)) dir.creat-e(output_ms5_clean, recursive = true)
    save(output, file = file.path(output_ms5_clean, files[i]))
  }
  
  # write csv files
  writepath <- file.path(outputdir, "output_beh/meta")
  if (!dir.exists(writepath)) dir.create(writepath, recursive = true)
  write.csv(removed, file.path(writepath, "excluded_nights.csv"), row.names = false)
  write.csv(removed_person, file.path(writepath, "excluded_nights_person.csv"), row.names = false)
  write.csv(datacleanmm, file.path(writepath, "dcleanmm.csv"), row.names = false)
  write.csv(datacleanww, file.path(writepath, "dcleanww.csv"), row.names = false)
  
  if (verbose) cat("post-processing completed for", r, "\n")
}

# function to re-run ggir part 5
re_run_ggir_part5 <- function(r, project_dir, project_deriv_dir, verbose=false) {
  subjectggirderiv <- function(x) {
    a <- dirname(x)
    output <- file.path(project_dir, project_deriv_dir, a)
    return(output)
  }
  
  outputdir <- subjectggirderiv(r)
  if (file.exists(file.path(outputdir, "ggircomplete.csv"))) {
    if (verbose) cat("ggir part 5 already re-run for", r, "\n")
    return()
  }
  
  datadir <- dirname(file.path(project_dir, r))
  writepath <- file.path(outputdir, "output_beh/meta")
  datacleanmmpath <- file.path(writepath, "dcleanmm.csv")
  datacleanwwpath <- file.path(writepath, "dcleanww.csv")
  metadatadir <- file.path(outputdir, "output_beh")
  
  trycatch({
    if (verbose) cat("re-running ggir part 5 for mm window for", r, "\n")
    g.shell.ggir(mode = 5,
                 metadatadir = metadatadir,
                 datadir = datadir,
                 outputdir = outputdir,
                 overwrite = true,
                 excludefirstlast.part5 = false,
                 threshold.lig = c(45), threshold.mod = c(100), threshold.vig = c(430),
                 boutdur.mvpa = c(1,5,10), boutdur.in = c(10,20,30), boutdur.lig = c(1,5,10),
                 boutcriter.mvpa=0.8,  boutcriter.in=0.9, boutcriter.lig=0.8,
                 timewindow=c("mm"),
                 acc.metric = "enmo",
                 data_cleaning_file=datacleanmmpath,
                 do.report = c(5),
                 visualreport = true,
                 do.parallel = true)
    
    if (verbose) cat("re-running ggir part 5 for ww window for", r, "\n")
    g.shell.ggir(mode = 5,
                 metadatadir = metadatadir,
                 datadir = datadir,
                 outputdir = outputdir,
                 overwrite = true,
                 excludefirstlast.part5 = false,
                 threshold.lig = c(45), threshold.mod = c(100), threshold.vig = c(430),
                 boutdur.mvpa = c(1,5,10), boutdur.in = c(10,20,30), boutdur.lig = c(1,5,10),
                 boutcriter.mvpa=0.8,  boutcriter.in=0.9, boutcriter.lig=0.8,
                 timewindow=c("ww"),
                 acc.metric = "enmo",
                 data_cleaning_file=datacleanwwpath,
                 do.report = c(5),
                 visualreport = true,
                 do.parallel = true)
    
    # create a completion file
    file.create(file.path(outputdir, "ggircomplete.csv"))
  }, error=function(e) {
    cat("error re-running ggir part 5 for", r, ":", e$message, "\n")
  })
}

# function to re-run intensity gradient calculations
re_run_intensity_gradient <- function(r, project_dir, project_deriv_dir, verbose=false) {
  subjectggirderiv <- function(x) {
    a <- dirname(x)
    output <- file.path(project_dir, project_deriv_dir, a)
    return(output)
  }
  
  outputdir <- subjectggirderiv(r)
  results_path <- file.path(outputdir, "output_beh/results/")
  if (file.exists(file.path(results_path, "part2_cleanedintensitygradient.csv"))) {
    if (verbose) cat("intensity gradient calculations already done for", r, "\n")
    return()
  }
  
  trycatch({
    part2_path <- file.path(results_path, "part2_daysummary.csv")
    part5_path <- file.path(results_path, "part5_daysummary_mm_l45m100v430_t5a5.csv")
    if (!file.exists(part2_path) || !file.exists(part5_path)) {
      if (verbose) cat("required files for intensity gradient not found for", r, "\n")
      return()
    }
    part2 <- read.csv(part2_path)
    part2 <- part2[c("filename","measurementday","ig_gradient_enmo_0.24hr","ig_intercept_enmo_0.24hr","ig_rsquared_enmo_0.24hr")]
    part5 <- read.csv(part5_path)
    part5 <- part5[c("window_number")]
    part2cleaned <- merge(part2, part5, by.x="measurementday", by.y="window_number")
    
    igpathday <- file.path(results_path, "part2_day_cleanedintensitygradient.csv")
    write.csv(part2cleaned, igpathday, row.names=false)
    
    part2cleanedperson <- data.frame(
      filename = unique(part2cleaned$filename),
      ndays = nrow(part2cleaned),
      ad_ig_gradient_enmo_0.24hr = mean(part2cleaned$ig_gradient_enmo_0.24hr),
      ad_ig_intercept_enmo_0.24hr = mean(part2cleaned$ig_intercept_enmo_0.24hr),
      ad_ig_ig_rsquared_enmo_0.24hr = mean(part2cleaned$ig_rsquared_enmo_0.24hr)
    )
    igpath <- file.path(results_path, "part2_person_cleanedintensitygradient.csv")
    write.csv(part2cleanedperson, igpath, row.names=false)
    if (verbose) cat("intensity gradient calculations completed for", r, "\n")
  }, error=function(e) {
    cat("error in intensity gradient calculations for", r, ":", e$message, "\n")
  })
}

# Function to process each path and create part 2-5 summaries
process_path <- function(path, base_output_dir) {
  # Set dir_path to the directory containing the raw data file
  dir_path <- dirname(path)
  
  # Print the dir_path for debugging
  print(paste("dir_path:", dir_path))
  
  # Extract subject and session from the path
  path_parts <- unlist(strsplit(path, "[/\\\\]+"))
  subject <- path_parts[grep("sub-GE", path_parts)][1]
  session <- path_parts[grep("ses-", path_parts)][1]
  
  # Ensure only ses-pre sessions are processed
  if (session != "ses-pre") {
    return()
  }
  
  # Set the output directory for the summaries
  output_dir <- file.path(base_output_dir, subject, session, "beh", "output_beh", "results")
  
  # Print the output_dir for debugging
  print(paste("output_dir:", output_dir))
  
  # Check if the directory exists and is writable
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    print(paste("Created directory:", output_dir))
  } else if (!file.access(output_dir, 2) == 0) {
    stop(paste("User does not have write access permissions for the directory:", output_dir))
  } else {
    print(paste("Directory exists and is writable:", output_dir))
  }
  
  # Print the variables before calling GGIR for debugging
  print(paste("Calling GGIR with dir_path:", dir_path, "and output_dir:", output_dir))
  
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

# Base output directory
base_output_dir <- "\\\\itf-rs-store24.hpc.uiowa.edu\\vosslabhpc\\Projects\\BETTER\\3-Experiment\\2-data\\bids\\derivatives\\GGIR_2.8.2"

# Project directory
project_dir <- "\\\\itf-rs-store24.hpc.uiowa.edu\\vosslabhpc\\Projects\\BETTER\\3-Experiment\\2-data\\bids"

# Loop through each subject directory and process the raw data in the beh folder
subject_dirs <- list.dirs(project_dir, recursive = FALSE, full.names = TRUE)
for (subject_dir in subject_dirs) {
  # Ensure the directory is a subject directory
  if (!grepl("sub-GE", subject_dir)) {
    next
  }
  
  # Construct the path to the ses-pre/beh folder
  ses_pre_folder <- file.path(subject_dir, "ses-pre", "beh")
  
  # Find the raw data file in the ses-pre/beh folder
  raw_files <- list.files(ses_pre_folder, pattern = "sub-GE.*_ses-pre_accel", full.names = TRUE)
  if (length(raw_files) == 0) {
    next
  }
  
  # Process each raw data file
  for (raw_file in raw_files) {
    process_path(raw_file, base_output_dir)
  }
}

print("All paths have been processed.")

# run the main function
main(opt)

