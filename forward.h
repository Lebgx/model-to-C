#pragma once
#ifndef FORWARD_H
#define FORWARD_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ------------------------�ṹ�嶨��-----------------------*/
/**
 * ��䷽ʽ
 * VALID���ڽ������ľ������֮ǰ����������������������
 * SAME��������������䷽ʽ��ʹ�������������������������������ͬ�ĳߴ�
 */
typedef enum pm
{
    VALID,
    SAME
}
padding_mode;



/**
 * @brief ��ʾ�����Ľṹ
 *
 * @param dims �����ĳߴ磺[d,h,w]
 * @param T ����������ֵ
 */
typedef struct {
    int dims[3];
    float*** T;
} Tensor;

/**
 * @brief ��ʾ�����Ľṹ
 *
 */
typedef struct {
    int n_kb;   // �����������Ҳ����bias_array�ĳ���
    int kernel_box_dims[3];     // ����˵ĳߴ磺[d,h,w]
    float**** kernel_box_group; // ����˵ľ���Ȩ����ֵ
    float* bias_array;          // ����˵�ƫ��ֵ
    int stride_x;   // x����Ĵ��ڲ���
    int stride_y;   // y����Ĵ��ڲ���
    padding_mode padding;    // �˾��������ѡ���padding_mode������
} ConvLayer;


/**
 * @brief ��ʾȫ���Ӳ�Ľṹ
 *
 */
typedef struct {
    int n_kb;   // �����������Ҳ����bias_array�ĳ���
    int kernel_box_dims[3];     // ����˵ĳߴ磺[d,h,w]
    float**** kernel_box_group; // ����˵ľ���Ȩ����ֵ
    float* bias_array;  // ����˵�ƫ��ֵ
} DenseLayer;

/* ---------------------------�����ɺ���-----------------------*/
// ��������Ȩ�صľ���㣬������ָ��ò��ָ��
ConvLayer* empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride_x, int stride_y, padding_mode padding);
// �������и���Ȩ�صľ���㣬������ָ��ò��ָ��
ConvLayer* new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float**** weights_array, float* biases_array, int stride_x, int stride_y, padding_mode padding);
// ����û��Ȩ�ص�ȫ���Ӳ㣬������ָ��ò��ָ��
DenseLayer* empty_Dense(int n_kb, int d_kb, int h_kb, int w_kb);
// �������и���Ȩ�ص�ȫ���Ӳ㣬������ָ��ò��ָ��
DenseLayer* new_Dense(int n_kb, int d_kb, int h_kb, int w_kb, float**** weights_array, float* biases_array);

/* -------------------------�������㺯��----------------------*/
// �������,��Ӧ�ø����ļ����֮ǰ��ͨ�������ľ�����ȡ����������
Tensor* Conv(Tensor* input, ConvLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input);
// ȫ��������,��Ӧ�ø����ļ����֮ǰ��ͨ���������ܼ����ȡ����������
Tensor* Dense(Tensor* input, DenseLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input);
// sigmoid�����
Tensor* sigmoid_activation(Tensor* input, int free_input);
// softmax�����
Tensor* softmax_activation(Tensor* input, int free_input);
// relu�����
Tensor* ReLU_activation(Tensor* input, int free_input);
// elu�����
Tensor* ELU_activation(Tensor* input, int free_input);
// ���Լ����
Tensor* linear_activation(Tensor* input, int free_input);
// �����Ӧ������������
Tensor* apply_padding(Tensor* input, int padding_x, int padding_y, int free_input);
// �ڸ���������������ִ�����ػ�
Tensor* MaxPool(Tensor* input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input);
// �ڸ���������������ִ�о�ֵ�ػ�
Tensor* AveragePool(Tensor* input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input);
// �ڸ���������������ִ���ϲ���
Tensor* UpSample(Tensor* input, int stride_x, int stride_y, int free_input);
// �ڸ���������������ִ��ƴ��
Tensor* Concatenate(Tensor* input1, Tensor* input2, int free_input);
// �ڸ���������������ִ������׼��
Tensor* BatchNormalization(Tensor* input, float** gamma, float** beta, float** moving_mean, float** moving_variance, int free_input);
// ����������չƽΪ���ȣ��Ա������Ⱥ͸߶�Ϊ1
Tensor* FlattenW(Tensor* input, int free_input);
// ����������չƽ����߶ȣ�ʹ�����ȺͿ��Ϊ1
Tensor* FlattenH(Tensor* input, int free_input);
// ����������չƽ������ȣ�ʹ����߶ȺͿ��Ϊ1��
Tensor* FlattenD(Tensor* input, int free_input);
// �������е���������Ԫ�����
Tensor* Add(Tensor* input1, Tensor* input2, int free_inputs);
// ���������� nD=1��2 άȫ�־�ֵ�ػ����������е���������Ԫ����ͺ󣬸ú�������������е�ÿ��Ԫ�س�����������n���õ�Ԫ�ؾ�ֵ
Tensor* GlobalAveragePooling(Tensor* input, int nD, int free_inputs);

/* -------------------------���ߺ���----------------------*/
// ��ӡ����
void print_tensor(Tensor* t);
// Ϊ����ά�ȣ�b*d*h*w����4D������������ڴ沢����ָ�롣
float**** alloc_4D(int b, int d, int h, int w);
// Ϊ����ά�ȣ�d*h*w����3D������������ڴ沢����ָ��
float*** alloc_3D(int d, int h, int w);
// ��ӡ�йظ�����������Ϣ
void print_conv_details(ConvLayer layer);
// ��ӡ�йظ���ȫ���Ӳ����Ϣ
void print_dense_details(DenseLayer layer);
// �ͷ�����tռ�õ��ڴ�
void free_tensor(Tensor* t);
// �����������µģ�d*h*w��ά������
Tensor* make_tensor(int d, int h, int w, float*** array);
// �ͷŷ���������������ڴ�ռ�
void free_ConvLayer(ConvLayer* layer);
// �ͷŷ��������ȫ���Ӳ���ڴ�ռ�
void free_DenseLayer(DenseLayer* layer);
#endif