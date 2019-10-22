# Realizability of planar point embeddings

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/duembgen/AngleRealizability/master)

This repository contains the code to reproduce all results of the 2020 ICASSP paper named

REALIZABILITY OF PLANAR POINT EMBEDDINGS FROM ANGLE MEASUREMENTS. 

## Authors

- Frederike Dümbgen 
- Majed El Helou
- Adam Scholefield

## Installation

This code has been tested with `python3.5.7`.

All requirements are available through pip and can be installed using 
```
pip install -r requirements.txt
```

## Use code

An example of how to use the code base is given in Analysis.ipynb.

## Reproduce figures

The Figures in the paper can be reproduced using the two below notebooks.

- Discrepancy.ipynb: Figures 2 and 3. 
- Angles_vs_Distances.ipynb: Figure 4.

These notebooks use pre-computed results.  To generate new results, you can run

- simulation_discrepancy.py: apply increasing number of constraints for denoising (used for Figure 3). Generates new results/discrepancy.pkl file.
- simulation_mds.py: angle vs. distance-based localization (used for Figure 4). Generates new results/angles_distances.pkl file.
