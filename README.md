# SiamRPN-PyTorch
Implementation SiamRPN on PyTorch with GOT-10k dataset  

## How to run Training
1. Download the GOT-10k dataset in http://got-10k.aitestunion.com/downloads
2. Run the train_siamrpn.py script:
```
cd train

python3 train_siamrpn.py --train_path=/path/to/dataset/GOT-10k/train
```

## How to run Tracking
[Coming Soon]


## pip install
```
pip3 install shapely
```

## How to fix GOT-10k dataset 

1. First you need to delete four videos:
```
GOT-10k_Train_008628 
GOT-10k_Train_008630 
GOT-10k_Train_009058  
GOT-10k_Train_009059
```
Because they are ymin and xmin is greater than the size of the image.

2. Run the fixedDataset.py script:
```
python3 fixedDataset.py --dataset_path=/path/to/dataset/GOT-10k/train
```



