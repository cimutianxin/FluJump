# FluJump科研项目
此项目名为 FluJump，是通过 HA 和其他元数据推断该流感是否会发生跨物种传播，也能预见能否从动物迁移到人类。

这是一个（seq+feature）2label 的预测任务，预测流感病毒是否会从物种1跳跃到物种2，以及跳跃的时间间隔。

## 环境
主项目使用conda管理环境。
FluJump的环境是env1，
其他baseline和消融环境实验需单独指定或询问。

## 数据
数据通过NCBI获取，预计需要2000年以后H1、H3、H5、H7四种亚型。
预期主要数据集列名主要包含[strain_name, HA_seq，多个标签，最初宿主，最初宿主时间，人类宿主时间，jump间隔]
数据的路径在data，数据下载和处理的代码在datascripts，

download_flu.py 是最初下载脚本，主要数据下载到data/raw

data/raw/H1～H7_raw.csv 是下载的原始数据

data/dataset_borkenhagen_raw 中是 Borkenhagen 的 Binding 数据

data/dataset_borkenhagen 是处理后的外部验证集

### 数据处理
Phase 1: 数据清洗
Phase 2: CD-HIT 序列聚类 99%
Phase 3: 宿主时间线与 Jump 标签
Phase 5: 还原成 isolate，使得每个样本都有 99% 聚类后的标签
Phase 6: MAFFT 亚型内对齐后整体对齐，得到 581 长度序列数据集，Borkenhagen baseline
Phase 7：外部验证数据集 Borkenhagen 的 binding 数据集

## 方法
ESM-2 + liner + mask_lf 多宿主分类

## baseline
1. ESM-2 + liner + normal_lf 多宿主分类
2. ESM-2 + liner 二分类
3. CNN 二分类
4. 

## 实验1
Binding 数据集上训练测试，ESM 主线*优于* CNN

## 实验2
H1H3 数据集训练验证测试，ESM 主线和 CNN *近似*
迁移到 H5 和 H7，ESM 主线*优于* CNN

## 实验2.1
迁移到 H5 和 H7，ESM 主线*优于* CNN，发现对于 H7，ESM 主线反向预测

## 实验3
假设该序列可以跨物种，ESM 主线可以通过 interval，推断跨物种能力分数

## 实验4
对 2000-2025 的样本进行
推断1: 是否可跨人，在2026数据中检查有没有出现。
推断2: interval预测，在2026及以后出现跨人的可能性。



## 消融实验
1. ESM-2 随机初始化参数 + liner + mask_lf 多宿主分类
2. ESM-2 + liner + normal_lf 多宿主分类
3. ESM-2 + ？+ mask_lf 多宿主分类

## 指标
AUC ACC

## 下游实验
跨物种 interval 预测

## 绘图
