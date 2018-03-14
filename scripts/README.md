## Example steps


#### Full example test

```
python -m pdb run_compiler_all_examples.py -e ../example-models/
```

The script caches the results after every processing 
every Stan example in ./test_compiler/status.pkl

Remove this file to re-run from scratch

Look at the dictionary called status and status_to_issue -- it can provide the files that
led to each error. Investigate each of them separately:

(1) stan2pyro <stan-file>

(2) python <generated-python-pyro-file> (test for syntax errors)

(3) Re-run the pipeline with cached status dictionary from that point -- 
to test the same file again after making relevant changes in 
the compiler/Python runtime


#### Convert data

```
Rscript --vanilla convert_data.R <R-data-file> <JSON-data-file>

Rscript --vanilla convert_data.R \
 ../example-models/bugs_examples/vol1/rats/rats.data.R \
 ../example-models/bugs_examples/vol1/rats/rats.data.json 
```


#### Generate pyro model code

```
../stan2pyro/bin/stan2pyro <stan-model-file> <pyro-model-file>

../stan2pyro/bin/stan2pyro \
 ../example-models/bugs_examples/vol1/rats/rats.stan \
 >  ../example-models/bugs_examples/vol1/rats/rats.pyro.py
```


#### Compare log probs from Stan and Pyro for this model

```
python compare_models.py -ds <JSON-data-file1> <JSON-data-file2> -s <stan-model-file> -p <pyro-model-file> -n <num_samples>

```

#### Comparison for "anchor" test models

```
python test_compare_models.py
```