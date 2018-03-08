## Example steps

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