# Welcome to Ben's NLP Tools

This script provides access to some basic NLP tools, like LIWC, Deidentification with Named Entity Recognition, using ChatGPT to rate text, and more.

## Installation

To install, complete the following steps:

0. If you want to use [LIWC](https://www.liwc.app), buy a license [here](https://www.liwc.app/buy)
1. If you haven't already download and install Anaconda. A simple way to do this is to download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
2. Download this repository and unzip it.
3. Open a terminal or VSCode and navigate to the directory you just unzipped.
4. Run the following command to recreate the environment: `conda env create -f environment.yml`
6. Copy the functions script `nlp_functions.py` to wherever you are working.
7. Activate the `nlp` conda environment in your IDE or Jupyter Notebook.
8. Import the functions script: `import nlp_functions as nlp`