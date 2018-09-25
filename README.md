# MEAutility

Python package for multi-electrode array (MEA) handling and stimulation.

## Installation

To install run `python setup.py install (or develop)`.
The package can then importd in Python `import MEAutility as MEA`

### Requirements
- numpy
- pyyaml
- matplotlib

## MEA definition

There are a few MEA models already installed (in the `MEAutility/electrodes` folder). MEa definition is done with a simple `yaml` file.

Here are a couple of examples:
`SqMEA-10-15um`:

```
electrode_name: SqMEA-10-15um
sortlist: null
dim: 10
pitch: 15
shape: square
size: 5
type: mea
```

- `electrode_name`: contains the MEA name
- `sortilist`: allows to add a channel map to re-order the electrodes
- `dim`: MEA dimensions. It can be a single value (for square MEA), a list of 2 values ([nrows, ncols]), or a list of N values in which each element i is the number of rows of column i.
- `pitch`: MEA pitch (inter-electrode distance). It can be a single value (for square MEA) or a list of 2 values ([pitch between rows, pitch between columns])
- `shape`: shape of each electrode. Either `circle` or `square`.
- `size`: size of each electrode. If the electrode is `circle` it is the redius, if `square` it is the half the electrode side.
- `type`: type of the MEA (only for plotting). It can be `mea` (for silicon probes) or `wire` (for microwires/tetrodes).
- `pos` (optional): list of 2D or 3D positions of each electrode.
- `plane` (optional): plane in which the electrode is instantiated (if `pos` is not given). It can be `xy`, `yz` (default), or `xz`.
- `model` (optional): modeling framework to compute stimulation potential. Either `inf` (infinite and conductive space), or `semi` (semi-infinite conductive space - the MEA is considered as an insulating plane).

Check out the `notebooks/electrode_definitions.ipynb` for other examples on commonly used probes (e.g. Neuropixels).

### Adding and Removing MEA models

The user can create new MEA models, e.g. `user-defined.yaml` or `user-defined.yml`, and add them to the package by:
```
MEA.add_mea('user-defined.yaml')
```
Available MEA models can be printed with: `MEA.return_mea()` and removed with `MEA.remove_mea('MEA-name')`.

## MEA instantiation and manipulation

First a MEA object needs to be instantiated:
```
mea = mea.return_mea('SqMEA-10-15um')
```
The positions and currents of each electrode can be accessed with `mea.positions` and `mea.currents`, respectively. The number of electrodes is stored in `mea.number_electrodes`.

Rotation and shift

## MEA stimulation

Each MEA contains a set of electrodes and each electrode has a `current` field. Currents are defined in [nA], positions in [um], and electric potential in [mV].
Several methods of the MEA class can be used to set currents. 

Now, with the previously created MEA, let's set the current of electrode 20 currents to 1 uA. This can be done in two ways:
```
mea.set_current(20, 1000) # 1000 nA = 1 uA
```
or 

```
currents = np.zeros(mea.number_electrodes)
currents[20] = 1000
mea.set_currents(currents)
```
We can define now some points in space and compute the electric potential generated by this set of currents:
```
# define 4 points
points = np.array([15, 20, 20],
                  [15, 20, 30],
                  [15, 20, 40],
                  [15, 20, 50])
```
The potential can be computed as:
```
v = mea.copute_field(points)
```
By default, each electrode is considered as a monopolar current source at the center of the elctrode. In order to consider the spatial extension of each stimulation point, one can set the field `points_per_electrode`. For example, if `mea.points_per_electrode = 100`, for each electrode 100 points are randomly drawn within the electrode boundaries and the current of that electrode is split among those points. Of course, this makes the computation of the potential slower, but more accurate (especially in proximity of the electrodes).

Currents can also have a temporal dynamics. `mea.currents` can be set to a n_elec x n_timepoints matrix. In this case, the `compute_fields` function returns an n_points x n_timepoints array.

## MEA plotting
