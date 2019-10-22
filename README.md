# Realizability of planar point embeddings

*It is normal that the below Binder link is not working as long as the repository
is not public yet. As soon as it is public, it should work.*

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/duembgen/AngleRealizability/master)

This repository contains the code to reproduce all results of the 2020 ICASSP paper named
*Realizability of planar point embeddings from angle measurements*.

```
@inproceedings{Duembgen2020,
  author={Duembgen Frederike and El Heloue Majed and Scholefield Adama}, 
  title={Realizability of planar point embeddings from angle measurements}, 
  booktitle={2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2020}, 
  pages={xxxx--xxxx}
}
```

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
