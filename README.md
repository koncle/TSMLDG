# TSMLDG
This is the source code of paper [https://openreview.net/forum?id=wb5nZa1TV](https://openreview.net/forum?id=wb5nZa1-TV).

## DataSets

The training procedure needs [gta5(G)](https://download.visinf.tu-darmstadt.de/data/from_games/), [synthia_rand_citys(S)](https://synthia-dataset.net/downloads/), [idd(I)](https://idd.insaan.iiit.ac.in/), [mapillary(M)](https://www.mapillary.com/dataset/vistas?pKey=1GyeWFxH_NPIQwgl0onILw) and [cityscapes(C)](https://www.cityscapes-dataset.com/). Please download them and put to the same folder which can be specified in `paths.py`

We also generated some synthetic images with [CycleGAN](https://github.com/junyanz/CycleGAN), we download the `style_ukyoe.pth`, `style_vangogh.pth`, `style_cezanne.pth` pretrained models and transfered GTA5 dataset with these three pretrained models. 

## Train the model

The model can be trained with `train.py`. If we want to train the MLDG with four source domains(G, S, I, M) and one target domain(C), we can parse these arguments.
```
python train.py --name exp --source GSIM --target C --train-num 1  
```
For more details, please refer to `train.py`

## Train the model

The model can be trained with `train.py`. If we want to train the MLDG with four source domains(G, S, I, M) and one target domain(C), we can parse these arguments.
```
python train.py --name exp --source GSIM --target C --train-num 1  
```
For more details, please refer to `train.py`

