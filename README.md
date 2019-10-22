# Realizability of planar point embeddings

This repository contains the code to reproduce all results of the 2020 ICASSP paper named

REALIZABILITY OF PLANAR POINT EMBEDDINGS FROM ANGLE MEASUREMENTS. 

## Installation

This code has been tested with `python3.5.7`.

All requirements are available through pip and can be installed using 
```
pip install -r requirements.txt

```

## Reproduce figures

- Discrepancy.ipynb: Figures 2 and 3. 
- Angles_vs_Distances.ipynb: Figure 4.

To generate new results, you can run
- simulation_discrepancy.py: apply increasing number of constraints for denoising (used for Figure 3). Generates new results/discrepancy.pkl file.
- simulation_mds.py: angle vs. distance-based localization (used for Figure 4). Generates new results/angles_distances.pkl file.
