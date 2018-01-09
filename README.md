[ADVI](https://arxiv.org/pdf/1506.03431.pdf)

[Pystan](https://pystan.readthedocs.io/en/latest/)

[Example Stan models](https://github.com/stan-dev/example-models)

## Compile C++ Stan2Pyro Script

NOTE: update the submodules

`git submodule update --init --recursive`

Generate makefile

`python makefile_gen.py > makefile`

compile using makefile

`make -j<num_processors>` e.g. `make -j15`

Example run:

`./stan2pyro/bin/stan2pyro <path/to/.stan_file>` e.g. `./stan2pyro/bin/stan2pyro ../../../pyro-main/examples/stan/example-models/bugs_examples/vol2/air/air.stan`
