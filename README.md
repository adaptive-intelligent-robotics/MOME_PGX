## Intro
This repository contains all code used in [Improving the Data Efficiency of Multi-Objective Quality-Diversity through Gradient Assistance and Crowding Exploration]() paper. This  builds on top of the [QDax framework](https://github.com/adaptive-intelligent-robotics/QDax) and includes the newly introduced _Multi-Objective Map-Elites with Policy Gradient assistance and crowding-based eXploration_ ([MOME-PGX](https://arxiv.org/abs/2302.12668)).


MOME-PGX is a Multi-Objective Quality-Diversity algorithm which is based on the [MOME algorithm](https://arxiv.org/abs/2202.03057). In particular, both algorithms extend the Quality-Diversity paradigm to multi-objective problems by maintaing a Pareto front in each cell of a MAP-Elites grid. However, MOME-PGX improves the performance and data-efficiency of MOME in two ways:

1) MOME-PGX uses gradient-based mutation operators for each objective function in order to quickly highlight promising regions of the solution space.
2) MOME-PGX uses crowding-based selection and addition mechanisms to promote exploration in all regions of the objective space.

<p align="center">
<img width="540" alt="teaser" src="https://user-images.githubusercontent.com/49594227/220638925-b67d335e-6c25-4af2-af2e-33e64226a3fe.png">
</p>

MOME-PGX is evaluated on four tasks from the [Brax Suite](https://pypi.org/project/brax/).

<p align="center">
<img width="653" alt="robots" src="https://user-images.githubusercontent.com/49594227/220644809-c4981384-85d7-485a-afcb-2cfec212a925.png">
</p>

In each task, the aim is to learn a variety of deep neural network controllers that enables the robot to walk while 1) maximising forward velocity and 2) minimising energy consumption. The behaviour descriptor of each controller is characterised as the proportion of time the robot spends on each foot.


## Installation

To run this code, you need to install the necessary libraries as listed in `requirements.txt` via:

```bash
pip install -r requirements.txt
```

However, we recommend using a containerised environment such as Docker, Singularity or conda  to use the repository. Further details are provided in the last section. 

## Basic API Usage

To run the MOME-PGX algorithm, or any other baseline algorithm mentioned in the paper, you just need to run the relevant `brax_[ALGO].py` script (where `[ALGO]` is the corresponding algorithm). For example, to run MOME-PGX, you can run:

```bash
python3 brax_mome_pgx.py
```

Or to run the MOME algorithm:
```bash
python3 brax_mome.py
```

The hyperparameters of the algorithms can be modified by changing their values in the `configs` directory of the repository. Alternatively, they can be modified directly in the command line. For example, to decrease the `pareto_front_max_length` parameter from 50 to 20 in MOME-PGX, you can run:

```bash
python3 brax_mome_pgx.py pareto_front_max_length=20
```

Running each algorithm automatically saves metrics, visualisations and plots of performance into a `results` directory. However, you can compare performance between algorithms once they have been run using the `analysis.py` script. To do this, you need to edit the list of the algorithms and environments you wish to compare and the metrics you wish to compute (at the bottom of `analysis.py`). Then, the relevant plots and performance metrics will be computed by running:

```bash
python3 analysis.py
```

## Singularity Usage

To build a final container (an executable file) using Singularity make sure you are in the root of the repository and then run:

```bash
singularity build --fakeroot --force singularity/[FINAL CONTAINER NAME].sif singularity/singularity.def
```

where you can replace '[FINAL CONTAINER NAME]' by your desired file name. When you get the final image, you can execute it via:

```bash
singularity -d run --app [APPNAME] --cleanenv --containall --no-home --nv [FINAL CONTAINER NAME].sif [EXTRA ARGUMENTS]
```

where 
- [FINAL CONTAINER NAME].sif is the final image built
- [APPNAME] is the name of the experiment you want to run, as specified by `%apprun` in the `singularity/singularity.def` file. There is a specific `%apprun` for each of the algorithms, ablations and baselines mentioned in the paper.
- [EXTRA ARGUMENTS] is a list of any futher arguments that you want to add. For example, you may want to change the random seed or Brax environment.


