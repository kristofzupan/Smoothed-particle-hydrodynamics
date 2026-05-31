# Smoothed Particle Hydrodynamics

SPH fluid simulation in Python. There are 4 different versions. The basic prototype implementation that only runs on the cpu is in the main file. The others are listed below and only run on a gpu due to using taichi library.

All the versions were tested and developed on a apple silicon m4 chip and are setup to use the Metal backend for Macs. `ti.init(arch=ti.metal)`

## Install

```
pip install -r requirements.txt
```

## Run

CPU version:
```
python main.py
```

GPU version (Metal backend - macOS):
```
python sph_gpu.py
```

3D Version (Metal backend - macOS):
```
python sph_3d.py
```

Marching cubes version (Metal backend - macOS):
```
python sph_mc.py
```