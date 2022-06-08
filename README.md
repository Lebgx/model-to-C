# model-to-C
## 用于将keras训练的model转为C源码

将keras训练的model转为C源码，使其摆脱Python环境及库依赖独立运行。

使用标准C实现，不依赖第三方库，不涉及硬件，便于在PC端或嵌入式设备中移植。

包括：
- 将model中权重数据和拓扑结构进行提取并自动转为C源码的脚本
- 标准C实现的一套前向传播计算库

## 注意
目前仅支持全连接层与卷积层计算。

keras输入格式要求为channel_last (height,width,channel)。

## 已实现：
### v0.23
- Windows环境下下采样、上采样函数的多线程实现

### v0.22
- Windows环境下各激活函数的多线程实现

### v0.21
- Windows环境下的Conv计算的多线程实现
- Windows环境下的Dense计算的多线程实现

### v0.1
- 层结构
  - 全连接层
  - Conv1D
  - Conv2D
- 计算函数
  - padding
  - UpSample
  - MaxPool
  - AveragePool
  - GlobalAveragePooling
  - Add
  - Concatenate
  - BatchNormalization
  - Flatten
  - 一系列常用激活函数

## 待实现：
- Windows环境下的多线程计算
- RNN
- 基于dgl的GNN
