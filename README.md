## Sample Code for our proposed SGA
SGA adapts a pretrained multilingual understanding model, XLMR, to a NAR generator in a parameter-efficient way.

### Implementation & Requirements
Our implementation is based on a public available codebase, https://github.com/THUNLP-MT/PLM4MT , which adapts a pretrained autoregressive model(mGPT) to translation tasks via multi-stage prompting (MSP). Please refer to https://github.com/THUNLP-MT/PLM4MT for detailed usage of codebase

In the sample code, we provide the model script, the data processing script, and the training script that differs from MSP, and we provide a simple train script for reference.

# Pipeline

## 集群

我对集群做出了一些修改，具体的修改可以通过 `vim ~/.bashrc` 查看。

TL;DR: 我加了很多 alias，同时自己的配置文件在 `/mnt/lustre/xujingjing/zhaochenyang/.bashrc` 下。可以查看我的 `alias`，都是常用命令。

## 环境

我从 python38 上 copy 了新的 conda 环境 CoT_MT，请不要混用，ovo

## 网络

集群的联网存在很大问题，可能得麻烦运维解决下。

## 数据集

英德数据集储存在 `/mnt/petrelfs/xujingjing/xujingjing/ted/de_en/`，请注意这个路径下的文件都是纯粹的文本文件，而不是二进制数据段。可以直接使用 `tf(tail -f)` 或者 `less` 查看。

按照 bohong 的处理需求，我将混合后的训练数据集存储在 ``/mnt/petrelfs/xujingjing/xujingjing/ted/de_en/train.merged`，而预处理代码位于 `/mnt/lustre/xujingjing/zhaochenyang/XLMR4MT/scripts/data_prepare.py`。

## 运行

参考 `/mnt/lustre/xujingjing/zhaochenyang/XLMR4MT/scripts/submit_train.sh` 即可。

## 训练

参考 `/mnt/lustre/xujingjing/zhaochenyang/XLMR4MT/scripts/train.sh`。

我删除了无关参数，并且修改了相关路径。

## 效果

大概每 1h 运行 2k step，存 1 个 ckpt，每个 ckpt 95M。 
