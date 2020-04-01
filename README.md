# Robustness Verification for Transformers

*Under construction*

This repo is for the code of our work:

Zhouxing Shi, Huan Zhang, Kai-Wei Chang, Minlie Huang, Cho-Jui Hsieh. [Robustness Verification for Transformers](https://openreview.net/pdf?id=BJxwPJHFwS). ICLR 2020.

## Train models

We first train models with different configurations on Yelp and SST-2 datasets:

```
./train_yelp.sh
./train_sst.sh
```

You may manually distribute the training runs in the scripts to different devices
for efficiency.

To evaluate the clean accuracy of the models:

```
python eval_accuracy.py
```

And the results will be saved to `res_acc.json`.

