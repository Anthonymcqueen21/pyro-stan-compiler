# Pyro Stan Compiler
Compiles [Stan](http://mc-stan.org/) models into [Pyro](http://pyro.ai) models.

## Usage

1. Clone the repo
```
git clone https://github.com/jpchen/pyro-stan-compiler.git
cd pyro-stan-compiler
git submodule update --init --recursive
```

2. Generate and run makefile
```
python makefile_gen.py > makefile
make -j [num-processors]
```

3. Run compiler
```
./stan2pyro/bin/stan2pyro <path/to/stan_file>
```

## Features

## Unsuported features
* `increment_log_prob`/`target` - manually manipulating the score in the ELBO is not supported in Pyro, though Pyro supports enumeration of discrete variables, so 
most models can be written as a `sample()` statement in Pyro.
* automatic vectorization - Pyro supports broadcasting and vectorization, which is not supported in Stan. Currently
for-loops are translated as-is into Pyro though they can be written in a vectorized manner for efficiency


## Models
