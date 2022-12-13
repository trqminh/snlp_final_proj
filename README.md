# snlp_final_proj
## Datasets and train models
```
this_repo/
  data/
    bbc_news/

  trained_models/
    bert_bbc_bestover50.pth
    ...
```
## To train
e.g.
```
python train.py --model distilbert --dataset bbc --epochs 50
```

## To test
e.g.
```
python test.py --model bert --dataset bbc --model_path trained_models/bert_bbc_bestover50.pth
```
