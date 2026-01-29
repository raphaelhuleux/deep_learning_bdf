# Deep Learning BDF

Course-style repo with notebooks and Python scripts that cover introductory neural networks and macroeconomic dynamic programming exercises.

## Structure

- 00-intro-nn: Intro neural network training notebooks (NumPy and PyTorch).
- 01-ramsey: solve the Ramsey model with deep-learning and compare to VFI
- 02-rbc: add TFP shocks to the Ramsey model (RBC mdoel), solve it with deep-learning and compare to VFI
- 03-consumption-saving: solve the simple consumption-saving model with idiosyncratic productivity shocks and a non-borrowing constraint with deep-learning and compare it to VFI

## Getting started

1. Create a Python environment (3.9+ recommended).
2. Install standard scientific packages plus PyTorch and QuantEcon.
	- Minimum set: numpy, scipy, matplotlib, pandas, jupyter, torch, quantecon.
	- Optional: seaborn for plots.
3. Install with either pip or conda:

	**pip**
	```bash
	pip install numpy scipy matplotlib pandas jupyter torch quantecon
	```

	**conda**
	```bash
	conda install numpy scipy matplotlib pandas jupyter
	conda install pytorch -c pytorch
	pip install quantecon
	```
3. Open the notebooks in Jupyter or VS Code and run cells top to bottom.

## Notes

- Scripts ending in `_vfi.py` implement value function iteration variants used by the notebooks.

## Acknowledgements 

- A key source of inspiration for all the code in this repository is Maliar, Maliar, Winant (2022) and the code [provided](https://github.com/QuantEcon/notebook-gallery/blob/22e3922ebe475f39a99a1df9bb953f42722eeaa1/ipynb/pablo_winant-dl_notebook.ipynb#L6). All remaining errors are naturally mine.
