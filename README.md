https://arxiv.org/pdf/1802.04537.pdf
https://yugeten.github.io/posts/2020/06/elbo/

# Build Steps

## Setting up Conda Environment

Install Conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

Create a new Conda environment for this project, based on the `environment.yml` file in the root directory.

```
$ conda env create -f environment.yml
```

Activate the environment.

```
$ conda activate elbo-comparison
```

## Updating Conda Environment with new dependencies

If you have installed any dependencies using the command `$ conda install ...`, you should export the updated environment file to `environment.yml`.

```
$ conda env export | tee environment.yml
```

A more succinct alternative is

```
$ conda env export --from-history | tee environment.yml
```
