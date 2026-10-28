# Code Repository: A Parallel City-Scale Acoustic Ground-Truth Engine for NLoS Embodied Perception

This repository contains the core C++ implementations of the physical engine presented in the main manuscript. 
---

## 📂 Repository Structure
```text
./
├── 01_acoustic_scattering.cpp          # Acoustic scattering physical model solver
├── 02_lloyds_mirror.cpp                # Lloyd's mirror interference effect simulation
├── 03_doppler_effect.cpp               # Doppler effect calculation module
├── 04_hpc_scalability.cpp              # High-performance parallel scalability benchmark
├── 05_sensor_illusion.cpp              # Sensor illusion (NLoS perception bias) simulation
├── 06_transient_localization_dx1.cpp   # Transient localization solver (dx=1.0 grid)
├── 07_transient_localization_dx0.5.cpp # Transient localization solver (dx=0.5 grid)
├── 08_city_scale_uav.cpp               # City-scale UAV acoustic propagation simulation
└── README.md                           # This instruction file