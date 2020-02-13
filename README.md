# Realizability of planar point embeddings

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/duembgen/AngleRealizability/master)

This repository contains the code to reproduce all results of the 2020 ICASSP paper named
*Realizability of planar point embeddings from angle measurements*.

```
@inproceedings{Duembgen2020,
  author={Dümbgen Frederike and El Helou Majed and Scholefield Adam}, 
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

This code base was developed for theoretical analysis and we do not guarantee efficiency or user-friendliness.
However, if the reader is interested in further development, the best starting points for learning how to use the code are the notebooks. 

## Reproduce figures

The Figures in the paper can be reproduced using the two below notebooks.

- `Analysis.ipynb`: Figure 2. 
- `Realizability.ipynb`: Figure 3. 
- `Angles_vs_Distances.ipynb`: Figure 4.

Figures 3 and 4 use pre-computed results.  To generate new results, you can run

- `simulation_discrepancy.py`: apply increasing number of constraints for denoising (used for Figure 3). 
- `simulation_angles_distances.py`: angle vs. distance-based localization (used for Figure 4). 
