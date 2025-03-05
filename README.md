# xenomorph
Modelling the morphology of Wolf-Rayet Colliding Wind Binary dust nebulae with fast and differentiable geometry.

![Apep](Images/Apep_evolution_pretty.gif)

## Installation
Currently the only way is to build from source. To do this, clone this repository and run
```
pip install .
```
when in the main `xenomorph` directory. We then recommend you create a file and run 
```python
import xenomorph as xm
```
to test that it built correctly.

This code uses Jax to speed up code runtime (through just-in-time compilation) and allows aspects of the code to be natively differentiable. This will be installed during the build process if not already in the environment.

## Documentation
I am still in the process of documenting the code itself, and also a readthedocs-like site. Check back here regularly to see what's going on. 

## Collaboration
If you run into any issues while using the code, please feel free to open an Issue or a Pull request. The code is, at present, more or less feature complete but there will be bugs hiding amongst the diamond I'm sure.
