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
Download datasets and trained models from [here](https://uark-my.sharepoint.com/:f:/g/personal/minht_uark_edu/EgnqPZOKMH5MgipwJd_1hfcBq5IXjeVrIY7fGBz24mVaFg?e=7q7X4J)
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
