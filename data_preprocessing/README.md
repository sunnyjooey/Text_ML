### Preparation 
Run in python to download all necessary files:  
info_extract.setupStanfordNLP()  

### Files
info_extract.py, stanford.py: downloads necessary files for text processing
process.py: process raw Tweets
parse.py: parse text into subject-verb-object
parallel_processing.py: script for multi-core parallel processing (use for parsing)
data_processing.ipynb: processes and pickles the data in prepration for feature generation
