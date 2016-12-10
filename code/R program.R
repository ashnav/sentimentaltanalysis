
#' Making several text files from a single CSV file
#' 
#' Convert a single CSV file (one text per row) into
#' separate text files. A function in R.
#' 
#' To use this function for the first time run:
#' install.packages("devtools")
#' then thereafter you just need to load the function 
#' fom github like so:
#' library(devtools) # windows users need Rtools installed, mac users need XCode installed
#' source_url("https://gist.github.com/benmarwick/9266072/raw/csv2txts.R")
#' 
#' Here's how to set the argument to the function
#' 
#' mydir is the full path of the folder that contains your csv file
#' for example "C:/Downloads/mycsvfile" Note that it must have 
#' quote marks around it and forward slashes, which are not default
#' in windows. This should be a folder with a *single* CSV file in it.
#' 
#' labels refers to the column number with the document labels. The 
#' default is the first column, but in can your want to use a different
#' column you can set it like so (for example if col 2 has the labels):
#' labels = 2
#' 
Author Preethi Hansaa,Ashwini murthy
Adapted from BenMarwick
Made few changes to make it compatible to accept twitter comma separated files
#' 
#' A full example, assuming you've sourced the 
#' function from github already:
#' csv2txt("C:/Downloads/mycsvfile", labels = 1)
#' and after a moment you'll get a message in the R console
#' saying 'Your texts files can be found in C:/Downloads/mycsvfile'


csv2txt <- function("C:/Users/windows/Desktop/R",labels = 1){

  mycsvfile <- list.files("C:/Users/windows/Desktop/R", full.names = TRUE, pattern = "*.CSV|.csv")
  
  mycsvdata <- read.csv(mycsvfile)
 
  mytxtsconcat <- apply(mycsvdata, 1, paste, collapse=" ")
  
  mytxtsdf <- data.frame(filename = mycsvdata[,1],fulltext = mytxtsconcat)
  

  
  setwd("C:/Users/windows/Desktop/NLP Proj")
  invisible(lapply(1:nrow(mytxtsdf), function(i) write.table(mytxtsdf[i,2], 
                                                               file = paste0(mytxtsdf[i,1], ".txt"),
                                                               row.names = FALSE, col.names = FALSE,
                                                               quote = FALSE)))
 
  message(paste0("Your text files can be found in ", getwd()))
}
