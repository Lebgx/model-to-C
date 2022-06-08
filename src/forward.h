#pragma once
#ifndef FORWARD_H
#define FORWARD_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>    // 用于PC端多线程计算

/* ------------------------结构体定义-----------------------*/
/**
 * 填充方式
 * VALID：不填充
 * SAME：输入张量的填充方式将使后续运算的输出具有与输入张量相同的尺寸
 */
typedef enum
{
    VALID,
    SAME
} padding_mode;

/**
 * @brief 表示张量的结构
 *
 * @param dims 张量的尺寸：[d,h,w]
 * @param T 张量具体数值
 */
typedef struct {
    int dims[3];
    float*** T;
} Tensor;

/**
 * @brief 表示卷积层的结构
 *
 */
typedef struct {
    int n_kb;   // 卷积核数量，也等于bias_array的长度
    int kernel_box_dims[3];     // 卷积核的尺寸：[d,h,w]
    float**** kernel_box_group; // 卷积核的具体权重数值
    float* bias_array;          // 卷积核的偏置值
    int stride_x;   // x方向的窗口步幅
    int stride_y;   // y方向的窗口步幅
    padding_mode padding;    // 此卷积层的填充选项，如padding_mode中所述
} ConvLayer;

/**
 * @brief 表示全连接层的结构
 *
 */
typedef struct {
    int n_kb;   // 卷积核数量，也等于bias_array的长度
    int kernel_box_dims[3];     // 卷积核的尺寸：[d,h,w]
    float**** kernel_box_group; // 卷积核的具体权重数值
    float* bias_array;  // 卷积核的偏置值
} DenseLayer;

/* ---------------------------层生成函数-----------------------*/
// 创建不带权重的卷积层，并返回指向该层的指针
ConvLayer* empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride_x, int stride_y, padding_mode padding);
// 创建具有给定权重的卷积层，并返回指向该层的指针
ConvLayer* new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float**** weights_array, float* biases_array, int stride_x, int stride_y, padding_mode padding);
// 创建没有权重的全连接层，并返回指向该层的指针
DenseLayer* empty_Dense(int n_kb, int d_kb, int h_kb, int w_kb);
// 创建具有给定权重的全连接层，并返回指向该层的指针
DenseLayer* new_Dense(int n_kb, int d_kb, int h_kb, int w_kb, float**** weights_array, float* biases_array);

/* -------------------------张量运算函数----------------------*/
// 卷积运算,在应用给定的激活函数之前，通过给定的卷积层获取给定的张量
Tensor* Conv(Tensor* input, ConvLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input);
// 全连接运算,在应用给定的激活函数之前，通过给定的密集层获取给定的张量
Tensor* Dense(Tensor* input, DenseLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input);
// sigmoid激活函数
Tensor* sigmoid_activation(Tensor* input, int free_input);
// softmax激活函数
Tensor* softmax_activation(Tensor* input, int free_input);
// relu激活函数
Tensor* ReLU_activation(Tensor* input, int free_input);
// elu激活函数
Tensor* ELU_activation(Tensor* input, int free_input);
// 线性激活函数
Tensor* linear_activation(Tensor* input, int free_input);
// 将填充应用于输入张量
Tensor* apply_padding(Tensor* input, int padding_x, int padding_y, int free_input);
// 在给定的输入张量上执行最大池化
Tensor* MaxPool(Tensor* input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input);
// 在给定的输入张量上执行均值池化
Tensor* AveragePool(Tensor* input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input);
// 在给定的输入张量上执行上采样
Tensor* UpSample(Tensor* input, int stride_x, int stride_y, int free_input);
// 在给定的输入张量上执行拼接
Tensor* Concatenate(Tensor* input1, Tensor* input2, int free_input);
// 在给定的输入张量上执行批标准化
Tensor* BatchNormalization(Tensor* input, float** gamma, float** beta, float** moving_mean, float** moving_variance, int free_input);
// 将输入张量展平为其宽度，以便输出深度和高度为1
Tensor* FlattenW(Tensor* input, int free_input);
// 将输入张量展平到其高度，使输出深度和宽度为1
Tensor* FlattenH(Tensor* input, int free_input);
// 将输入张量展平到其深度，使输出高度和宽度为1。
Tensor* FlattenD(Tensor* input, int free_input);
// 对数组中的张量进行元素求和
Tensor* Add(Tensor* input1, Tensor* input2, int free_inputs);
// 对张量进行 nD=1或2 维全局均值池化，对数组中的张量进行元素求和后，该函数将结果数组中的每个元素除以张量数量n，得到元素均值
Tensor* GlobalAveragePooling(Tensor* input, int nD, int free_inputs);

/* -------------------------上述函数的多线程实现----------------------*/
/**
 * @brief 用于传递数据给calc_conv_t线程函数的结构体
 *
 */
typedef struct {
    // 指向计算所需数据
    Tensor* input;
    ConvLayer* layer;
    // 计算边界
    int output_d;
    int output_h;
    int output_w;
    // 存放当前进度，全局共享
    int *d; 
    int *h;
    int *w;
    // 存放各线程的计算结果，全局共享
    float*** output_array;
} Struct_Conv_T;
/**
 * @brief 用于传递数据给calc_dense_t线程函数的结构体
 *
 */
typedef struct {
    // 指向计算所需数据
    Tensor* input;
    DenseLayer* layer;
    // 计算边界
    int output_w;
    // 存放当前进度，全局共享
    int* w;
    // 存放各线程的计算结果，全局共享
    float*** output_array;
} Struct_Dense_T;

/**
 * @brief 用于传递数据给calc_activation_t线程函数的结构体
 *
 */
typedef struct {
    // 指向计算所需数据
    Tensor* input;
    // 存放计算边界、各线程的计算结果，全局共享
    Tensor* output;
    // 存放当前进度，全局共享
    int* d;
    int* h;
    int* w;
    float* sum; // 记录softmax的加和操作累计值，全局共享
    int* isAdd;  // 判断softmax的加和操作是否完成，全局共享
} Struct_Activation_T;
/**
 * @brief 用于传递数据给calc_pool_t线程函数的结构体
 *
 */
typedef struct {
    // 指向计算所需数据
    Tensor* input;
    // 计算边界
    int output_d;
    int output_h;
    int output_w;
    // 参数
    int stride_x;
    int stride_y;
    int height;
    int width;
    // 存放当前进度，全局共享
    int* d;
    int* h;
    int* w;
    // 存放各线程的计算结果，全局共享
    float*** output_array;
} Struct_Pool_T;
/**
 * @brief 用于传递数据给calc_upsample_t线程函数的结构体
 *
 */
typedef struct {
    // 指向计算所需数据
    Tensor* input;
    // 计算边界
    int output_d;
    int output_h;
    int output_w;
    // 参数
    int stride_x;
    int stride_y;
    // 存放当前进度，全局共享
    int* d;
    int* h;
    int* w;
    // 存放各线程的计算结果，全局共享
    float*** output_array;
} Struct_UpSample_T;

// 卷积运算【多线程实现】
Tensor* Conv_t(Tensor* input, ConvLayer* layer, void* (*calc_activation_t)(void*), int free_input, int num_of_thread);
// 每个线程的conv具体计算函数
void* calc_conv_t(void* args);
// 全连接运算【多线程实现】
Tensor* Dense_t(Tensor* input, DenseLayer* layer, void* (*calc_activation_t)(void*), int free_input, int num_of_thread);
// 每个线程的dense具体计算函数
void* calc_dense_t(void* args);
// 激活函数运算【多线程实现】
Tensor* Activation_t(Tensor* input, void* (*calc_activation_t)(void*), int free_input, int num_of_thread);
// 每个线程的sigmoid具体计算函数
void* calc_sigmoid_t(void* args);
// 每个线程的softmax具体计算函数
void* calc_softmax_t(void* args);
// 每个线程的ReLU具体计算函数
void* calc_ReLU_t(void* args);
// 每个线程的ELU具体计算函数
void* calc_ELU_t(void* args);
// 每个线程的linear具体计算函数
void* calc_linear_t(void* args);
// 池化运算【多线程实现】
Tensor* Pool_t(Tensor* input, void* (*calc_activation_t)(void*), int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input, int num_of_thread);
// 每个线程的maxpool具体计算函数
void* calc_maxpool_t(void* args);
// 每个线程的averagepool具体计算函数
void* calc_averagepool_t(void* args);
// 上采样运算【多线程实现】
Tensor* UpSample_t(Tensor* input, int stride_x, int stride_y, int free_input, int num_of_thread);
// 每个线程的upsample具体计算函数
void* calc_upsample_t(void* args);




/* -------------------------工具函数----------------------*/
// 打印张量
void print_tensor(Tensor* t);
// 为具有维度（b*d*h*w）的4D浮点数组分配内存并返回指针。
float**** alloc_4D(int b, int d, int h, int w);
// 为具有维度（d*h*w）的3D浮点数组分配内存并返回指针
float*** alloc_3D(int d, int h, int w);
// 打印有关给定卷积层的信息
void print_conv_details(ConvLayer layer);
// 打印有关给定全连接层的信息
void print_dense_details(DenseLayer layer);
// 释放张量t占用的内存
void free_tensor(Tensor* t);
// 创建并配置新的（d*h*w）维度张量
Tensor* make_tensor(int d, int h, int w, float*** array);
// 释放分配给给定卷积层的内存空间
void free_ConvLayer(ConvLayer* layer);
// 释放分配给给定全连接层的内存空间
void free_DenseLayer(DenseLayer* layer);
#endif