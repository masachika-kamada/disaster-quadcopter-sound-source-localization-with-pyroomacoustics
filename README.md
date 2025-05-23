# Disaster Quadcopter Sound Source Localization with PyRoomAcoustics

This repository is for simulating the localization of survivors by voice in disaster sites using a quadcopter. It allows for the recording of multiple sound sources such as survivors' voices, environmental sounds, and the quadcopter's ego-noise using a microphone array mounted on the quadcopter, based on PyRoomAcoustics. It is possible to set various parameters such as the microphone array, quadcopter, sound sources, and terrain. The recorded sound can also be used for sound source localization with multiple MUSIC-based algorithms.

## Reference

If you use this repository, please cite the following paper:

Masachika Kamada, Junji Yamato, Yasuhiro Oikawa, Hiroshi G. Okuno, Jun Ohya,  
"Locating Survivors’ Voices in Disaster Sites Using Quadcopters Based on Modeling Complicated Environments by PyRoomAcoustics and SSL by MUSIC-based Algorithms,"  
in *2025 IEEE/SICE International Symposium on System Integration (SII)*, pp. 846-853, 2025.  
[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10871001) | DOI: 10.1109/SII.2025.10871001

### BibTeX

```bibtex
@inproceedings{kamada2025locating,
  title={Locating Survivors’ Voices in Disaster Sites Using Quadcopters Based on Modeling Complicated Environments by PyRoomAcoustics and SSL by MUSIC-based Algorithms},
  author={Kamada, Masachika and Yamato, Junji and Oikawa, Yasuhiro and Okuno, Hiroshi G and Ohya, Jun},
  booktitle={2025 IEEE/SICE International Symposium on System Integration (SII)},
  pages={846--853},
  year={2025},
  organization={IEEE},
  doi={10.1109/SII.2025.10871001}
}
```

## Usage

### 1. Environment Setup

Create an execution environment from [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code.

### 2. Recording Sounds through Simulation

Edit `experiments/config.yaml` to set the desired parameters.
```
python generate_acoustic_sim.py
```

### 3. Sound Source Localization with MUSIC Algorithms

```
python compute_doa.py
```

## Directory Structure

```
.
├── data/
│   ├── ambient/
│   ├── drone/
│   ├── snr/
│   └── voice/
├── experiments/
│   ├── data/
│   └── config.yaml
├── lib/
│   ├── doa/
│   └── custom/
├── notebooks/
│   ├── impulse_response_measured/
│   └── ...
├── src/
├── compute_doa.py
└── generate_acoustic_sim.py
```

* `data/`: Contains audio data to be placed in the simulation environment.
* `experiments/`: Stores settings and results for each experiment.
  * `data/`: Directory where results from simulations and sound source localization are saved.
* `lib/`: Custom external libraries.
* `notebooks/`: Stores Jupyter notebooks.
  * `impulse_response_measured/`: Data on impulse responses from PyRoomAcoustics and real environments.
* `src`: Contains modules and packages used in the project.
