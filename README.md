# 浮世絵作者予測
浮世絵作者予測 Competition LB3位(学生LB1位)ソースコード

## 学習の流れ

0. 準備  
学習・テスト用データをダウンロードして`input`ディレクトリに入れる．Jupyter Notebookを起動して[`visualize_and_preprocess.ipynb`](https://github.com/tattaka/ukiyoe/blob/master/src/visualize_and_preprocess.ipynb)を開き`Run All`してください．  
LINE Notify APIを用いて学習終了時に通知を飛ばすようにしているので[`train.py`](https://github.com/tattaka/ukiyoe/blob/master/src/train.py#L153)や[`train_pseudo_label.py`](https://github.com/tattaka/ukiyoe/blob/master/src/train_pseudo_label.py#L177)の該当部分を必要に応じて削除してください．このcallbackを使いたい人は環境変数`LINE_TOKEN`を設定すると使えると思います．

1. First stage learning
``` shell
$ cd src
$ python train.py --config cbamresnext50_mish_jpunet_cfg
$ python make_submission.py --config cbamresnext50_mish_jpunet_cfg
$ python train.py --config densenet121_mish_jpunet_cfg
$ python make_submission.py --config densenet121_mish_jpunet_cfg
$ python train.py --config resnet50_mish_jpunet_cfg
$ python make_submission.py --config resnet50_mish_jpunet_cfg
$ python train.py --config inceptionresnetv2_mish_jpunet_cfg
$ python make_submission.py --config inceptionresnetv2_mish_jpunet_cfg
$ python train.py --config seresnext50_mish_jpunet_cfg
$ python make_submission.py --config seresnext50_mish_jpunet_cfg
$ python make_stacking.py
```
実行することでCV0.90~0.91が，`predicts/stacking_tta.csv`を提出することでLB0.912が得られます．

2. Second stage learning
``` shell
$ python train.py --config densenet121_mish_jpunet_pl_cfg
$ python make_submission.py --config densenet121_mish_jpunet_pl_cfg
```
実行することでCV0.91~0.92が，`JPUNet_densenet121_diffrgrad_pl_256x256.csv`を提出することでLB0.927が得られます．

## 動作確認ができている実行環境
### PC環境
* Ubuntu 16.04.6 LTS(Docker)
* GeForce GTX 1080 Ti x3
* Nvidia-Driver==430.40
* CUDA Version==10.1(Docker)
* Python==3.6.4(Docker)

### pip packages
* numpy==1.17.2
* pandas==0.25.1
* sklearn==0.21.3
* Pillow==6.1.0
* opencv-python==4.1.1.26
* scipy==1.3.1
* torch==1.2.0
* albumentations==0.4.3
* catalyst==19.9.5
* efficientnet-pytorch==0.5.1
* pretrainedmodels==0.7.4
* pytorch-toolbelt==0.2.1
* tensorboard==2.0.0
* tensorboardX==1.8
* tqdm==4.36.1
