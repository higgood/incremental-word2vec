# incremental-word2vec
Modify word2vec such that it's possible to "condition" on existing embeddings for some words, and induce embeddings for new words.


# usage:
```
./word2vec -train testdemo.txt -output testdemo.oldmodel -size 200 -threads 12
./word2vec -train new_data.txt -output testdemo.newmodel -size 200 -threads 12 -fixed-embeddings testdemo.oldmodel
```
