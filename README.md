# KPNet

Official pytorch implementation of the KPNet

# TODO

* [X] 🎉️ Submit source code
* [ ] Overview
* [ ] Submit implementation details
* [ ] Perfect link to the paper
* [ ] Citation

* [X] 🎉️  Acknowledgment
* [X] 🎉️ Getting Started
* [ ] Reference

# Overview


# Implementation details


# Getting Started


1. [Dataset](#1)
2. [Configure](#2)
3. [Test](#3)
4. [Train](#4)
5. [Metrics](#5)

## <span id="1">Dataset

we conduct experiment on  **HDRTV** [3] and HM16.9. This dataset contains 1235 paired training pictures and 117 test pictures.Please refer to the paper for the details on the processing of this dataset. This dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1NEE22Pty_n_u5Esdv9lcqA?pwd=vdrzhttps://pan.baidu.com/s/1YfdFYD03KMyhKnpDo9nnZw)(accsee code: vdrz)

## <span id="2">Configure


## <span id="3">How to test

we provide the pretrained models to test, which can be downloaded from the link provided by the dataset. Please put the obtained files into models according to the instructions.

* Modify the pictures and model paths in the *.yml file(`cd code/options/test/**.` )
* ```bash
  cd code/
  python /test.py -opt /options/test/**/test.yml 
  ```

## <span id="4">How to train

* Prepare the data.
* make sure that the paths and settings in `./options/train/**` are correct,then run

```bash
cd  code

python train.py -opt /option/train/**.yml
```

## <span id="5">Metrics

Can be found in the metrics folder

# Reference

1. X. Chen, Z. Zhang, J. Ren, L. Tian, Y. Qiao, and C. Dong. A new journey from SDRTV to HDRTV. In  *ICCV* ,
   pages 4500–4509, 2021.

# Citation
