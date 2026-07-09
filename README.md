# FluJump科研项目
此项目名为FluJump，是通过HA和其他元数据推断该流感是否会发生跨物种传播，也能预见能否从动物迁移到人类。

这是一个（seq+feature）2label的预测任务，预测流感病毒是否会从物种1跳跃到物种2，以及跳跃的时间间隔。

## 环境
主项目使用conda管理环境。
FluJump的环境是env1，
其他baseline和消融环境实验需单独指定或询问。

## 数据
数据通过NCBI获取，预计需要2000年以后H1、H3、H5、H7四种亚型。
预期主要数据集列名主要包含[strain_name, HA_seq，多个标签，最初宿主，最初宿主时间，人类宿主时间，jump间隔]
数据的路径在data，数据下载和处理的代码在datascripts，

### 最初数据
download_flu.py是最初下载脚本，主要数据下载到data/raw

data/raw/H1～H7_raw.csv 是下载的原始数据

### 数据处理
Phase 1: 数据清洗
Phase 2: CD-HIT 序列聚类
Phase 3: 宿主时间线与 Jump 标签
Phase 4: 特征工程


## 方法

## baseline

## 消融实验

## 指标

## 下游实验

## 绘图

## Skills