# Learning leaves a memory trace in motor cortex - Supporting code

This repository contains code supporting the findings and experiments presented in the paper ["Learning leaves a memory trace in motor cortex"](https://www.cell.com/current-biology/abstract/S0960-9822(24)00298-7) by Losey et al. (2024). Published in Current Biology.

## Citation
Losey, Darby M., et al. "Learning leaves a memory trace in motor cortex." Current Biology 34.7 (2024): 1519-1531.

## Contents
- `mt_example_code.ipynb`: A Jupyter Notebook to exemplify the analyses conducted in the manuscript.
- `mt_functions.py`: A Python module containing functions called by `mt_example_code.ipynb`. This module encapsulates the core algorithms and data manipulation processes used in our research.

## Prerequisites
This code pack relies on standard Python libraries. Specific versions included below, however, it will likely run
without issue on any versions, provided they are not too outdated.

Library versions used in the drafting of the manuscript:
- Python 3.11 (https://www.python.org/)
- Pandas 2.0.3 (https://pandas.pydata.org/)
- NumPy 1.24.3 (https://numpy.org/)
- Matplotlib 3.7.2 (https://matplotlib.org/)
- SciPy 1.11.1 (https://www.scipy.org/)

Examples showecased using:
- Jupyter Notebook 6.5.4 (https://jupyter.org/)

For detailed installation instructions, refer to the official documentation of the respective libraries.

## Data
Example data is contained in the data folder in pickle files (https://docs.python.org/3/library/pickle.html). 

## Installation
1. Clone this repository to your local machine using `git clone`
2. Navigate to the repository's directory.
3. Install the required Python libraries using `pip`: pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 scipy==1.11.1 notebook==6.5.4

This command will install the specific versions of the dependencies listed above.

## Running the Notebook

To run the `mt_example_code.ipynb` notebook:
1. Launch Jupyter Notebook in the repository's directory.
2. Open `mt_example_code.ipynb`.
3. Execute the cells in sequence to observe the data processing and analysis steps.
