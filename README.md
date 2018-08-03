# R<sup>3</sup>Net: Recurrent Residual Refinement Network for Saliency Detection

by Zijun Deng, Xiaowei Hu, Lei Zhu, Xuemiao Xu, Jing Qin, Guoqiang Han, and Pheng-Ann Heng [[paper link](https://www.ijcai.org/proceedings/2018/95)]

This implementation is written by Zijun Deng at the South China University of Technology.

***

## Citation
@inproceedings{deng18r,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Deng, Zijun and Hu, Xiaowei and Zhu, Lei and Xu, Xuemiao and Qin, Jing and Han, Guoqiang and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {R$^{3}${N}et: Recurrent Residual Refinement Network for Saliency Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {IJCAI},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}

## Saliency Map
The results of salienct object detection on five datasets (ECSSD, HKU-IS, PASCAL-S, SOD, DUT-OMRON) can be found 
at [Google Drive](https://drive.google.com/open?id=1PloaTokZEfWPy8voDm7mp3yvHnXCtn2c).

## Trained Model
You can download the trained model which is reported in our paper at 
[Google Drive](https://drive.google.com/open?id=1Y50Cj5Ek-ZIsFj03_pRMSsvqXXeIJSaS).

## Requirement
* Python 2.7
* PyTorch 0.4.0
* torchvision
* numpy
* Cython
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Training
1. Set the path of pretrained ResNeXt model in resnext/config.py
2. Set the path of MSRA10K dataset in config.py
3. Run by ```python train.py```

The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.

*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently 
change them as you need.

Training a model on a single GTX 1080Ti GPU takes about 70 minutes.

## Testing
1. Set the path of five benchmark datasets in config.py
2. Put the trained model in ckpt/R3Net
2. Run by ```python infer.py```

*Settings* of testing were gathered at the beginning of *infer.py* and you can conveniently 
change them as you need.

## Useful links
* [MSRA10K](http://mmcheng.net/msra10k/): our training set
* [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), 
[HKU-IS](https://sites.google.com/site/ligb86/hkuis), 
[PASCAL-S](http://cbi.gatech.edu/salobj/), 
[SOD](http://elderlab.yorku.ca/SOD/), 
[DUT-OMRON](http://ice.dlut.edu.cn/lu/DUT-OMRON/Homepage.htm): the five benchmark datasets
