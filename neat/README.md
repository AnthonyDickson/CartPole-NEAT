# NEAT Genetic Algorithm
In this project I have decided to implement NEAT (NeuroEvolution of augmenting topologies), a genetic reinforcement learning algorithm, to solve the cart pole problem. The NEAT Implementation project deals with all the machine learning related stuff (as opposed to the web-based visualization stuff).

## Getting Started
1.  You can create the required python environment and install the required dependencies with [conda](https://conda.io/docs/user-guide/install/index.html):
    ```shell
    $ conda env create -f environment.yml
    ```

2.  You can run the program with the command:
    ```shell
    $ python -m neat
    ```
    You can add the flag '--help' to see a list of options.

## Unit Tests
All tests can be run with the following command:
```shell
$ python -m tests
```
and individual tests can run as such:
```shell
$ python -m tests.graph
```
where 'graph' can be replaced with test you want to run.
