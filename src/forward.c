#include "forward.h"

/**
 * @brief   ��������Ȩ�صľ���㣬������ָ��ò��ָ�롣
 *
 * @param n_kb      ����˵���������Ҳ��ƫ�������
 * @param d_kb      ����˵����
 * @param h_kb      ����˵ĸ߶�
 * @param w_kb      ����˵Ŀ��
 * @param stride_x  x���򲽳�
 * @param stride_y  y���򲽳�
 * @param padding   ��䷽ʽ��VALID/SAME
 * @return ConvLayer*
 */
ConvLayer* empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride_x, int stride_y, padding_mode padding)
{
    ConvLayer* convolution_layer_pointer;
    convolution_layer_pointer = malloc(sizeof(ConvLayer));

    convolution_layer_pointer->n_kb = n_kb;
    convolution_layer_pointer->kernel_box_dims[0] = d_kb;
    convolution_layer_pointer->kernel_box_dims[1] = h_kb;
    convolution_layer_pointer->kernel_box_dims[2] = w_kb;

    convolution_layer_pointer->kernel_box_group = alloc_4D(n_kb, d_kb, h_kb, w_kb);
    convolution_layer_pointer->bias_array = malloc(n_kb * sizeof(float));

    convolution_layer_pointer->stride_x = stride_x;
    convolution_layer_pointer->stride_y = stride_y;
    convolution_layer_pointer->padding = padding;

    return convolution_layer_pointer;
}

/**
 * @brief   �������и���Ȩ�صľ���㣬������ָ��ò��ָ��
 *
 * @param n_kb          ����˵���������Ҳ��ƫ�������
 * @param d_kb          ����˵����
 * @param h_kb          ����˵ĸ߶�
 * @param w_kb          ����˵Ŀ��
 * @param weights_array ���������Ȩ�ص�4D�������飨n_kb * d_kb * h_kb * w_kb��
 * @param biases_array  ����Ϊn_kb�ĸ���ƫ������
 * @param stride_x      x���򲽳�
 * @param stride_y      y���򲽳�
 * @param padding       ��䷽ʽ��VALID/SAME
 * @return ConvLayer*
 */
ConvLayer* new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float**** weights_array, float* biases_array, int stride_x, int stride_y, padding_mode padding)
{
    ConvLayer* convolution_layer_pointer;
    convolution_layer_pointer = malloc(sizeof(ConvLayer));

    convolution_layer_pointer->n_kb = n_kb;
    convolution_layer_pointer->kernel_box_dims[0] = d_kb;
    convolution_layer_pointer->kernel_box_dims[1] = h_kb;
    convolution_layer_pointer->kernel_box_dims[2] = w_kb;

    convolution_layer_pointer->kernel_box_group = weights_array;
    convolution_layer_pointer->bias_array = biases_array;

    convolution_layer_pointer->stride_x = stride_x;
    convolution_layer_pointer->stride_y = stride_y;
    convolution_layer_pointer->padding = padding;

    return convolution_layer_pointer;
}

/**
 * @brief   ����û��Ȩ�ص�ȫ���Ӳ㣬������ָ��ò��ָ��
 *
 * @param n_kb  ����˵���������Ҳ��ƫ�������
 * @param d_kb  ����˵����
 * @param h_kb  ����˵ĸ߶�
 * @param w_kb  ����˵Ŀ��
 * @return DenseLayer*
 */
DenseLayer* empty_Dense(int n_kb, int d_kb, int h_kb, int w_kb)
{
    DenseLayer* dense_layer_pointer;
    dense_layer_pointer = malloc(sizeof(DenseLayer));

    dense_layer_pointer->n_kb = n_kb;
    dense_layer_pointer->kernel_box_dims[0] = d_kb;
    dense_layer_pointer->kernel_box_dims[1] = h_kb;
    dense_layer_pointer->kernel_box_dims[2] = w_kb;

    dense_layer_pointer->kernel_box_group = alloc_4D(n_kb, d_kb, h_kb, w_kb);
    dense_layer_pointer->bias_array = malloc(n_kb * sizeof(float));

    return dense_layer_pointer;
}

/**
 * @brief   �������и���Ȩ�ص�ȫ���Ӳ㣬������ָ��ò��ָ��
 *
 * @param n_kb          ����˵���������Ҳ��ƫ�������
 * @param d_kb          ����˵����
 * @param h_kb          ����˵ĸ߶�
 * @param w_kb          ����˵Ŀ��
 * @param weights_array ���������Ȩ�ص�4D�������飨n_kb * d_kb * h_kb * w_kb��
 * @param biases_array  ����Ϊn_kb�ĸ���ƫ������
 * @return DenseLayer*
 */
DenseLayer* new_Dense(int n_kb, int d_kb, int h_kb, int w_kb, float**** weights_array, float* biases_array)
{
    DenseLayer* dense_layer_pointer;
    dense_layer_pointer = malloc(sizeof(DenseLayer));

    dense_layer_pointer->n_kb = n_kb;
    dense_layer_pointer->kernel_box_dims[0] = d_kb;
    dense_layer_pointer->kernel_box_dims[1] = h_kb;
    dense_layer_pointer->kernel_box_dims[2] = w_kb;

    dense_layer_pointer->kernel_box_group = weights_array;
    dense_layer_pointer->bias_array = biases_array;

    return dense_layer_pointer;
}

/**
 * @brief   �������,��Ӧ�ø����ļ����֮ǰ��ͨ�������ľ�����ȡ����������
 *
 * @param input         ��������
 * @param layer         �����
 * @param activation    ָ�򼤻���ĺ���ָ��
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* Conv(Tensor* input, ConvLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input)
{
    if (input->dims[0] != layer->kernel_box_dims[0])
    {
        printf("Error���˲㣨 % d�����ں˿����ȼ������������� % d������ȱ���ƥ��\n", layer->kernel_box_dims[0], input->dims[0]);
        exit(EXIT_FAILURE);
    }

    if (layer->padding == SAME)
    {
        int padding_x, padding_y;
        padding_x = layer->stride_x * (input->dims[2] - 1) - input->dims[2] + layer->kernel_box_dims[2]; // left + right
        padding_y = layer->stride_y * (input->dims[1] - 1) - input->dims[1] + layer->kernel_box_dims[1]; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // ���padding����ʹ'input'ָ��ԭʼ����ĸ��������ͷ�'input'�ǰ�ȫ��
    }

    int output_d = layer->n_kb;
    int output_w, output_h;

    // ��ʽ�е�����ʡ�ԣ���Ϊ��ʱ�����������Ѿ������,��ߴ�Ҳ��Ӧ����
    output_h = ((input->dims[1] /*+ 2*layer->padding */ - layer->kernel_box_dims[1]) / layer->stride_y) + 1;
    output_w = ((input->dims[2] /*+ 2*layer->padding */ - layer->kernel_box_dims[2]) / layer->stride_x) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, id, by, bx, i, j;

    // ����������飬�������ÿ����Ԫ���ֵ
    for (d = 0; d < output_d; d++)
    { // ������
        for (h = 0; h < output_h; h++)
        { // ����߶�
            for (w = 0; w < output_w; w++)
            {                              // ������
                output_array[d][h][w] = 0; // ���ڼ�¼����������ÿ����ͨ�����ϵľ��֮��
                for (id = 0; id < input->dims[0]; id++)
                {                             // �������
                    by = h * layer->stride_y; //"begin y" �����ں˴��ڵ��ϱ�Ե��������ϵ�λ��
                    bx = w * layer->stride_x; //"begin x" �����ں˴������Ե��������ϵ�λ��
                    for (i = 0; i < (layer->kernel_box_dims[1]); i++)
                    { // �����ں˴��ڵĸ߶�
                        for (j = 0; j < (layer->kernel_box_dims[2]); j++)
                        { // �����ں˴��ڵĿ��
                            output_array[d][h][w] += input->T[id][by + i][bx + j] * layer->kernel_box_group[d][id][i][j];
                        }
                    }
                }
                // ���ƫ��
                output_array[d][h][w] += layer->bias_array[d];
            }
        }
    }

    if (free_input)
        free_tensor(input);

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    return activation(output, 1);
}

/**
 * @brief   ȫ��������,��Ӧ�ø����ļ����֮ǰ��ͨ���������ܼ����ȡ����������
 *
 * @param input         ��������
 * @param layer         ȫ���Ӳ�
 * @param activation    ָ�򼤻���ĺ���ָ��
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* Dense(Tensor* input, DenseLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input)
{
    if (input->dims[0] != layer->kernel_box_dims[0] || input->dims[1] != layer->kernel_box_dims[1] || input->dims[2] != layer->kernel_box_dims[2])
    {
        printf("Error��ά�Ȳ�ƥ��\n");
        printf("���룺��d:%d h:%d w:%d ��| ����ˣ��� d:%d h:%d w:%d��\n", input->dims[0], input->dims[1], input->dims[2], layer->kernel_box_dims[0], layer->kernel_box_dims[1], layer->kernel_box_dims[2]);
        exit(EXIT_FAILURE);
    }

    int output_d = 1, output_h = 1;
    int output_w = layer->n_kb;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, id, i, j;

    // �����������������飬�������ÿ����Ԫ���ֵ
    for (d = 0; d < output_d; d++)
    { // ������
        for (h = 0; h < output_h; h++)
        { // ����߶�
            for (w = 0; w < output_w; w++)
            { // ������
                output_array[d][h][w] = 0;
                for (id = 0; id < input->dims[0]; id++)
                { // ������ȣ�����ȫ���Ӳ�ͨ��Ϊ1����Ϊ����֮ǰͨ�������չƽ����
                    for (i = 0; i < layer->kernel_box_dims[1]; i++)
                    { // �����ں˴��ڵĸ߶�
                        for (j = 0; j < layer->kernel_box_dims[2]; j++)
                        { // �����ں˴��ڵĿ��
                            output_array[d][h][w] += input->T[id][i][j] * layer->kernel_box_group[w][id][i][j];
                        } // ����by��bx����0������ı䣬��Ϊ�ں�ά�ȵ�������������ά��
                    }
                }

                // ���ƫ��
                output_array[d][h][w] += layer->bias_array[w];
            }
        }
    }

    if (free_input)
        free_tensor(input);

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    return activation(output, 1);
}

/**
 * @brief sigmoid�����
 *
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* sigmoid_activation(Tensor* input, int free_input)
{
    Tensor* output;
    int d, h, w;

    if (free_input)
    {
        output = input;
    }
    else
    {
        float*** output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for (d = 0; d < output->dims[0]; d++)
    {
        for (h = 0; h < output->dims[1]; h++)
        {
            for (w = 0; w < output->dims[2]; w++)
            {
                output->T[d][h][w] = ((float)(1 / (1 + exp((double)-1 * (input->T[d][h][w])))));
            }
        }
    }

    return output;
}

/**
 * @brief softmax�����
 * y=e^(xi) / ( e^(x1) + e^(x2) + ...  e^(xn) )
 * 
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* softmax_activation(Tensor* input, int free_input)
{
    Tensor* output;
    int d, h, w;

    if (free_input)
    {
        output = input;
    }
    else
    {
        float*** output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    float sum = 0;
    for (d = 0; d < output->dims[0]; d++)
    {
        for (h = 0; h < output->dims[1]; h++)
        {
            for (w = 0; w < output->dims[2]; w++)
            {
                sum += exp(input->T[d][h][w]);
            }
        }
    }

    for (d = 0; d < output->dims[0]; d++)
    {
        for (h = 0; h < output->dims[1]; h++)
        {
            for (w = 0; w < output->dims[2]; w++)
            {
                output->T[d][h][w] = exp((float)(input->T[d][h][w])) / sum;
            }
        }
    }

    return output;
}

/**
 * @brief relu�����
 *
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* ReLU_activation(Tensor* input, int free_input)
{
    Tensor* output;
    int d, h, w;

    if (free_input)
    {
        output = input;
    }
    else
    {
        float*** output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for (d = 0; d < output->dims[0]; d++)
    {
        for (h = 0; h < output->dims[1]; h++)
        {
            for (w = 0; w < output->dims[2]; w++)
            {
                output->T[d][h][w] = (input->T[d][h][w] < 0) ? 0 : input->T[d][h][w];
            }
        }
    }

    return output;
}

/**
 * @brief elu�����
 *
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* ELU_activation(Tensor* input, int free_input)
{
    Tensor* output;
    int d, h, w;

    if (free_input)
    {
        output = input;
    }
    else
    {
        float*** output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for (d = 0; d < output->dims[0]; d++)
    {
        for (h = 0; h < output->dims[1]; h++)
        {
            for (w = 0; w < output->dims[2]; w++)
            {
                output->T[d][h][w] = (input->T[d][h][w] < 0) ? ((float)exp(input->T[d][h][w]) - 1) : input->T[d][h][w];
            }
        }
    }

    return output;
}

/**
 * @brief ���Լ����
 *
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* linear_activation(Tensor* input, int free_input)
{
    if (free_input)
        return input;

    Tensor* output;
    int d, h, w;

    float*** output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
    output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);

    for (d = 0; d < output->dims[0]; d++)
    {
        for (h = 0; h < output->dims[1]; h++)
        {
            for (w = 0; w < output->dims[2]; w++)
            {
                output->T[d][h][w] = input->T[d][h][w];
            }
        }
    }

    return output;
}

/**
 * @brief   �����Ӧ������������
 * �����Ҫ��ͬ�����(�����ڲ����������Գ�����ǲ����ܵ�)����ô�����������ѭtensorflow���ʵ�֣�tensor�ĵײ����Ҳཫ��ö�������
 * 
 * @param input         ��������
 * @param padding_x     �������+���������������������
 * @param padding_y     �������+���������������������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* apply_padding(Tensor* input, int padding_x, int padding_y, int free_input)
{
    int output_d = input->dims[0];
    int output_h = input->dims[1] + padding_y;
    int output_w = input->dims[2] + padding_x;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, x, y, squeeze_along_x, squeeze_along_y;

    for (d = 0; d < output_d; d++)
    {
        // ���¶Գ����
        for (squeeze_along_y = 0; squeeze_along_y < (padding_y / 2); squeeze_along_y++)
        {
            for (x = 0; x < output_w; x++)
            {
                output_array[d][squeeze_along_y][x] = output_array[d][(output_h - 1) - squeeze_along_y][x] = 0;
            }
        }

        // �����Ƿ�Գ�
        if (padding_y % 2)
        {
            // ��������Ǹ���ȱ�������ŵײ�������Щ0��
            for (x = 0; x < output_w; x++)
            {
                output_array[d][(output_h - 1) - (padding_y / 2)][x] = 0;
            }
        }

        // ���ҶԳ����
        for (squeeze_along_x = 0; squeeze_along_x < (padding_x / 2); squeeze_along_x++)
        {
            for (y = 0; y < output_h; y++)
            {
                output_array[d][y][squeeze_along_x] = output_array[d][y][(output_w - 1) - squeeze_along_x] = 0;
            }
        }

        // �����Ƿ�Գ�
        if (padding_x % 2)
        {
            // ��������Ǹ���ȱ������������������Щ0��
            for (y = 0; y < output_h; y++)
            {
                output_array[d][y][(output_w - 1) - (padding_x / 2)] = 0;
            }
        }

        // ���м��ԭ��������
        for (x = (padding_x / 2); x < (output_w - (padding_x / 2) - (padding_x % 2)); x++)
        {
            for (y = (padding_y / 2); y < (output_h - (padding_y / 2) - (padding_y % 2)); y++)
            {
                output_array[d][y][x] = input->T[d][y - (padding_y / 2)][x - (padding_x / 2)];
            }
        }
    }

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief   �ڸ���������������ִ�����ػ�
 *
 * @param input         ��������
 * @param height        ���ڵĸ߶�
 * @param width         ���ڵĿ��
 * @param stride_x      x����Ĵ��ڲ���
 * @param stride_y      y����Ĵ��ڲ���
 * @param padding       ��䷽ʽ
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* MaxPool(Tensor* input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input)
{
    if (padding == SAME)
    {
        int padding_x, padding_y;
        padding_x = stride_x * (input->dims[2] - 1) - input->dims[2] + width;  // left + right
        padding_y = stride_y * (input->dims[1] - 1) - input->dims[1] + height; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // ���������ʹ'input'ָ��ԭʼ����ĸ��������ͷ�'input'�ǰ�ȫ��
    }

    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = ((input->dims[2] - width) / stride_x) + 1;
    output_h = ((input->dims[1] - height) / stride_y) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, i, j, by, bx;
    float max;

    // ����������飬�������ÿ����Ԫ���ֵ
    for (d = 0; d < output_d; d++)
    { // ������
        for (h = 0; h < output_h; h++)
        { // ����߶�
            for (w = 0; w < output_w; w++)
            { //  ������
                by = h * stride_y;
                bx = w * stride_x;
                max = input->T[d][by][bx];
                for (i = 0; i < height; i++)
                { // �����������ڵĸ߶�
                    for (j = 0; j < width; j++)
                    { // �����������ڵĿ��
                        if ((input->T[d][by + i][bx + j]) > max)
                        {
                            max = input->T[d][by + i][bx + j];
                        }
                    }
                }
                output_array[d][h][w] = max;
            }
        }
    }

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);
    return output;
}

/**
 * @brief    �ڸ���������������ִ�о�ֵ�ػ�
 *
 * @param input         ��������
 * @param height        ���ڵĸ߶�
 * @param width         ���ڵĿ��
 * @param stride_x      x����Ĵ��ڲ���
 * @param stride_y      y����Ĵ��ڲ���
 * @param padding       ��䷽ʽ
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* AveragePool(Tensor* input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input)
{
    if (padding == SAME)
    {
        int padding_x, padding_y;
        padding_x = stride_x * (input->dims[2] - 1) - input->dims[2] + width;  // left + right
        padding_y = stride_y * (input->dims[1] - 1) - input->dims[1] + height; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // ���������ʹ'input'ָ��ԭʼ����ĸ��������ͷ�'input'�ǰ�ȫ��
    }

    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = ((input->dims[2] - width) / stride_x) + 1;
    output_h = ((input->dims[1] - height) / stride_y) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, i, j, by, bx;
    float ave;

    // ����������飬�������ÿ����Ԫ���ֵ
    for (d = 0; d < output_d; d++)
    { // ������
        for (h = 0; h < output_h; h++)
        { // ����߶�
            for (w = 0; w < output_w; w++)
            { //  ������
                by = h * stride_y;
                bx = w * stride_x;
                ave = 0;
                for (i = 0; i < height; i++)
                { // �����������ڵĸ߶�
                    for (j = 0; j < width; j++)
                    { // �����������ڵĿ��
                        ave += input->T[d][by + i][bx + j];
                    }
                }
                output_array[d][h][w] = ave / (height * width);
            }
        }
    }

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);
    return output;
}

/**
 * @brief �ڸ���������������ִ���ϲ���
 * (d,x,y)->(d,m*x,n*y)
 *
 * @param input         ��������
 * @param stride_x      x������ϲ�������
 * @param stride_y      y������ϲ�������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* UpSample(Tensor* input, int stride_x, int stride_y, int free_input)
{
    int output_d = input->dims[0];
    int output_x = input->dims[1] * stride_x;
    int output_y = input->dims[2] * stride_y;
    float*** output_array = alloc_3D(output_d, output_x, output_y);

    int d, x, y, i, j;
    // ����������飬�������ÿ����Ԫ���ֵ
    for (d = 0; d < output_d; d++) // ������
    {
        for (x = 0; x < input->dims[1]; x++) // ����߶�
        {
            for (y = 0; y < input->dims[2]; y++) // ����߶�
            {
                for (i = 0; i < stride_x; i++)
                {
                    for (j = 0; j < stride_y; j++)
                    {
                        output_array[d][stride_x * x + i][stride_y * y + j] = input->T[d][x][y];
                    }
                }
            }
        }
    }

    Tensor* output;
    output = make_tensor(output_d, output_x, output_y, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief �ڸ���������������ִ��ƴ��
 * (d1, x, y)+(d2, x, y) -> (d1+d2, x, y)
 *
 * @param input1 ��������1
 * @param input2 ��������2
 * @return Tensor*
 */
Tensor* Concatenate(Tensor* input1, Tensor* input2, int free_input)
{
    if (input1->dims[1] != input2->dims[1] || input1->dims[2] != input2->dims[2])
    {
        printf("Error: ������h��w��һ�£��޷�ƴ�ӡ�\n");
        exit(EXIT_FAILURE);
    }

    int d1, d2, x, y;
    d1 = input1->dims[0];
    d2 = input2->dims[0];
    x = input1->dims[1];
    y = input1->dims[2];

    int output_d = d1 + d2;

    float*** output_array = alloc_3D(output_d, x, y);
    int i, j;
    for (i = 0; i < d1; i++)
    {
        for (j = 0; j < x; j++)
        {
            output_array[i][j] = input1->T[i][j];
        }
    }

    for (i = d1; i < output_d; i++)
    {
        for (j = 0; j < x; j++)
        {
            output_array[i][j] = input2->T[i - d1][j];
        }
    }

    Tensor* output;
    output = make_tensor(output_d, x, y, output_array);

    if (free_input)
    {
        free_tensor(input1);
        free_tensor(input2);
    }
    return output;
}

/**
 * @brief �ڸ���������������ִ������׼��
 * y = gamma(x-moving_mean)/sqrt(moving_variance+��)+beta
 * 
 * @param input             ��������
 * @param gamma             ����ϵ��
 * @param beta              ƫ��ϵ��
 * @param moving_mean       ��ֵϵ��
 * @param moving_variance   ����ϵ��
 * @param free_input        �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* BatchNormalization(Tensor* input, float** gamma, float** beta, float** moving_mean, float** moving_variance, int free_input)
{
    int output_d = input->dims[0];
    int output_h = input->dims[1];
    int output_w = input->dims[2];
    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w;
    // ����������飬�������ÿ����Ԫ���ֵ

    for (d = 0; d < output_d; d++) // �������
    {
        for (h = 0; h < output_h; h++)
        {
            for (w = 0; w < output_w; w++)
            {
                output_array[d][h][w] = gamma[0][d] * ((input->T[d][h][w] - moving_mean[0][d]) / (sqrt(moving_variance[0][d]) + 0.001)) + beta[0][d];
            }
        }
    }

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief   ����������չƽΪ���ȣ��Ա������Ⱥ͸߶�Ϊ1��(d,h,w) -> (1,1,d*h*w)
 *
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* FlattenW(Tensor* input, int free_input)
{
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_d = 1, output_h = 1;
    int output_w = input_d * input_h * input_w;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, i = 0;

    // keras��shapeΪ��h,w,d��,���ⶨ���ʽΪ��d,h,w����kerasĬ��Ϊ��h,w,d��������ֱ��̯ƽ
    /*for (w = 0; w < output_w; w++) {
        output_array[0][0][w] = input->T[w / (input_w * input_h)][(w / input_w) % input_h][w % input_w];
    }*/

    for (w = 0; w < input_w; w++)
    {
        for (h = 0; h < input_h; h++)
        {
            for (d = 0; d < input_d; d++)
            {
                output_array[0][0][i++] = input->T[d][h][w];
            }
        }
    }

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief   ����������չƽ����߶ȣ�ʹ�����ȺͿ��Ϊ1��(d,h,w) -> (1,d*h*w,1)
 *
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* FlattenH(Tensor* input, int free_input)
{
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_d = 1, output_w = 1;
    int output_h = input_d * input_h * input_w;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int h;

    for (h = 0; h < output_h; h++)
    {
        output_array[0][h][0] = input->T[h / (input_h * input_w)][(h / input_w) % input_h][h % input_w];
    }

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief   ����������չƽ������ȣ�ʹ����߶ȺͿ��Ϊ1��(d,h,w) -> (d*h*w,1,1)
 *
 * @param input         ��������
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* FlattenD(Tensor* input, int free_input)
{
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_w = 1, output_h = 1;
    int output_d = input_d * input_h * input_w;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d;

    for (d = 0; d < output_d; d++)
    {
        output_array[d][0][0] = input->T[d / (input_h * input_w)][(d / input_w) % input_h][d % input_w];
    }

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief   �������е�����������Ԫ�����
 *
 * @param input1        ����1
 * @param input2        ����2
 * @param free_inputs   �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* Add(Tensor* input1, Tensor* input2, int free_inputs)
{

    if (input1->dims[0] != input2->dims[0] || input1->dims[1] != input2->dims[1] || input1->dims[2] != input2->dims[2])
    {
        printf("Error: ������d��h��w��һ�£��޷���ӡ�\n");
        exit(EXIT_FAILURE);
    }

    Tensor* output;
    float*** output_array = alloc_3D(input1->dims[0], input1->dims[1], input1->dims[2]);
    output = make_tensor(input1->dims[0], input1->dims[1], input1->dims[2], output_array);

    int d, h, w;
    for (d = 0; d < output->dims[0]; d++)
    {
        for (h = 0; h < output->dims[1]; h++)
        {
            for (w = 0; w < output->dims[2]; w++)
            {
                output->T[d][h][w] = input1->T[d][h][w] + input2->T[d][h][w];
            }
        }
    }

    if (free_inputs)
    {
        free_tensor(input1);
        free_tensor(input2);
    }

    return output;
}
/**
 * @brief   ����������ĳά�Ƚ���Ԫ����ͺ󣬳��Ը�ά�ȵĳ���n���õ�Ԫ�ؾ�ֵ
 *
 * @param input         ����
 * @param nD            ��ֵ�ػ���ά�ȣ�1��2
 * @param free_inputs   �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* GlobalAveragePooling(Tensor* input, int nD, int free_inputs)
{
    if (nD != 1 && nD != 2)
    {
        printf("Error��dimsӦΪ1��2\n");
        exit(EXIT_FAILURE);
    }
    Tensor* output;
    int d, h, w;

    if (nD == 1)
    { // GlobalAveragePooling1D   (d,1,n)->(1,1,d)
        float*** output_array = alloc_3D(input->dims[1], 1, input->dims[0]);
        output = make_tensor(input->dims[1], 1, input->dims[0], output_array);

        for (d = 0; d < output->dims[2]; d++)
        {
            for (h = 0; h < output->dims[0]; h++)
            {
                output->T[h][0][d] = 0;
                for (w = 0; w < input->dims[2]; w++)
                {
                    output->T[h][0][d] += input->T[d][h][w];
                }
                output->T[h][0][d] /= input->dims[2];
            }
        }
    }
    else
    { // GlobalAveragePooling2D   (d,m,n)->(1,1,d)
        float*** output_array = alloc_3D(1, 1, input->dims[0]);
        output = make_tensor(1, 1, input->dims[0], output_array);

        for (d = 0; d < output->dims[2]; d++)
        {
            output->T[0][0][d] = 0;
            for (h = 0; h < input->dims[1]; h++)
            {
                for (w = 0; w < input->dims[2]; w++)
                {
                    output->T[0][0][d] += input->T[d][h][w];
                }
            }
            output->T[0][0][d] /= (input->dims[1] * input->dims[2]);
        }
    }

    if (free_inputs)
    {
        free_tensor(input);
    }

    return output;
}

/**
 * @brief   ���ڴ�ӡ����������̨
 *
 * @param t ����ӡ������ָ��
 */
void print_tensor(Tensor* t)
{

    printf("ά�ȣ�%d,%d,%d\n\n---------------------------------------\n", t->dims[0], t->dims[1], t->dims[2]);

    int i, j, k;
    for (i = 0; i < t->dims[0]; i++)
    {
        printf("\nLayer %d:\n\n", i);
        for (j = 0; j < t->dims[1]; j++)
        {
            for (k = 0; k < t->dims[2]; k++)
            {
                printf("%f ", t->T[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

/**
 * @brief   Ϊ����ά�ȣ�b*d*h*w����4D������������ڴ沢����ָ�롣
 *
 * @param b     ά�� 0, size of float *** array
 * @param d     ά�� 1, size of float ** array
 * @param h     ά�� 2, size of float * array
 * @param w     ά�� 3, size of float array
 * @return float****
 */
float**** alloc_4D(int b, int d, int h, int w)
{
    float**** newArr;
    newArr = malloc(b * sizeof(float***));
    if (newArr == NULL)
    {
        printf("Error: �޷�Ϊalloc_4D.newArr�����ڴ�\n");
        exit(EXIT_FAILURE);
    }

    int i, j, k;
    for (i = 0; i < b; i++)
    {
        newArr[i] = malloc(d * sizeof(float**));
        if (newArr[i] == NULL)
        {
            printf("Error: �޷�Ϊalloc_4D.newArr[%d]�����ڴ�\n", i);
            exit(EXIT_FAILURE);
        }
        for (j = 0; j < d; j++)
        {
            newArr[i][j] = malloc(h * sizeof(float*));
            if (newArr[i][j] == NULL)
            {
                printf("Error: �޷�Ϊalloc_4D.new[%d][%d]�����ڴ�\n", i, j);
                exit(EXIT_FAILURE);
            }
            for (k = 0; k < h; k++)
            {
                newArr[i][j][k] = malloc(w * sizeof(float));
                if (newArr[i][j][k] == NULL)
                {
                    printf("Error: �޷�Ϊalloc_4D.new[%d][%d][%d]�����ڴ�\n", i, j, k);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    return newArr;
}

/**
 * @brief   Ϊ����ά�ȣ�d*h*w����3D������������ڴ沢����ָ�롣
 *
 * @param d     ά�� 0, size of float ** array
 * @param h     ά�� 1, size of float * array
 * @param w     ά�� 2, size of float array
 * @return float***
 */
float*** alloc_3D(int d, int h, int w)
{
    float*** newArr;
    newArr = malloc(d * sizeof(float**));
    if (newArr == NULL)
    {
        printf("Error: �޷�Ϊalloc_3D.newArr�����ڴ�\n");
        exit(EXIT_FAILURE);
    }

    int i, j;
    for (i = 0; i < d; i++)
    {
        newArr[i] = malloc(h * sizeof(float*));
        if (newArr[i] == NULL)
        {
            printf("Error: �޷�Ϊalloc_3D.newArr[%d]�����ڴ�\n", i);
            exit(EXIT_FAILURE);
        }
        for (j = 0; j < h; j++)
        {
            newArr[i][j] = malloc(w * sizeof(float));
            if (newArr[i][j] == NULL)
            {
                printf("Error: �޷�Ϊalloc_3D.newArr[%d][%d]�����ڴ�\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    return newArr;
}

/**
 * @brief ��ӡ�йظ�����������Ϣ
 *
 * @param layer ����ӡ�ľ����
 */
void print_conv_details(ConvLayer layer)
{
    printf("�����λ�ڣ�%p\n", &layer);
    printf("\t�����������%d\n", layer.n_kb);
    printf("\t�����ά�ȣ�%d,%d,%d\n", layer.kernel_box_dims[0], layer.kernel_box_dims[1], layer.kernel_box_dims[2]);
    printf("\tx������%d\n", layer.stride_x);
    printf("\ty������%d\n", layer.stride_y);
    printf("\t��䷽ʽ��%d\n\n", layer.padding);

    int n, d, h, w;
    printf("\tȨ����Ϣ��\n");
    for (n = 0; n < layer.n_kb; n++)
    {
        printf("\tBox %d:\n", n);
        for (d = 0; d < layer.kernel_box_dims[0]; d++)
        {
            printf("\t\tLayer %d:\n", d);
            for (h = 0; h < layer.kernel_box_dims[1]; h++)
            {
                for (w = 0; w < layer.kernel_box_dims[2]; w++)
                {
                    printf("\t\t\t%f ", layer.kernel_box_group[n][d][h][w]);
                }
                printf("\n");
            }
        }
    }
    printf("\tƫ����Ϣ��\n");
    for (n = 0; n < layer.n_kb; n++)
    {
        printf("\t\t%f\n", layer.bias_array[n]);
    }
}

/**
 * @brief ��ӡ�йظ���ȫ���Ӳ����Ϣ
 * 
 * @param layer ����ӡ�ľ����
 */
void print_dense_details(DenseLayer layer)
{
    printf("�����λ�ڣ�%p\n", &layer);
    printf("\t�����������%d\n", layer.n_kb);
    printf("\t�����ά�ȣ�%d,%d,%d\n", layer.kernel_box_dims[0], layer.kernel_box_dims[1], layer.kernel_box_dims[2]);

    int n, d, h, w;
    printf("\tȨ����Ϣ��\n");
    for (n = 0; n < layer.n_kb; n++)
    {
        printf("\tBox %d:\n", n);
        for (d = 0; d < layer.kernel_box_dims[0]; d++)
        {
            printf("\t\tLayer %d:\n", d);
            for (h = 0; h < layer.kernel_box_dims[1]; h++)
            {
                for (w = 0; w < layer.kernel_box_dims[2]; w++)
                {
                    printf("\t\t\t%f ", layer.kernel_box_group[n][d][h][w]);
                }
                printf("\n");
            }
        }
    }
    printf("\tƫ����Ϣ��\n");
    for (n = 0; n < layer.n_kb; n++)
    {
        printf("\t\t%f\n", layer.bias_array[n]);
    }
}

/**
 * @brief �ͷ�����tռ�õ��ڴ�
 *
 * @param t ���ͷ��ڴ������
 */
void free_tensor(Tensor* t)
{
    int d, h;
    for (d = 0; d < t->dims[0]; d++)
    {
        for (h = 0; h < t->dims[1]; h++)
        {
            free(t->T[d][h]);
        }
        free(t->T[d]);
    }
    free(t->T);
    free(t);
}

/**
 * @brief       �����������µģ�d*h*w��ά������
 *
 * @param d     �������
 * @param h     �����߶�
 * @param w     �������
 * @param array ����ά�ȣ�d*h*w����3D�������飬�����й�������
 * @return Tensor*
 */
Tensor* make_tensor(int d, int h, int w, float*** array)
{
    Tensor* new_tensor;
    new_tensor = malloc(sizeof(Tensor));
    new_tensor->T = array;
    new_tensor->dims[0] = d;
    new_tensor->dims[1] = h;
    new_tensor->dims[2] = w;

    return new_tensor;
}

/**
 * @brief �ͷŷ���������������ڴ�ռ�
 *
 * @param layer ���ͷŵľ����
 */
void free_ConvLayer(ConvLayer* layer)
{
    int n, d, h;
    for (n = 0; n < layer->n_kb; n++)
    {
        for (d = 0; d < layer->kernel_box_dims[0]; d++)
        {
            for (h = 0; h < layer->kernel_box_dims[1]; h++)
            {
                free(layer->kernel_box_group[n][d][h]);
            }
            free(layer->kernel_box_group[n][d]);
        }
        free(layer->kernel_box_group[n]);
    }
    free(layer->kernel_box_group);
    free(layer->bias_array);
    free(layer);
}

/**
 * @brief �ͷŷ��������ȫ���Ӳ���ڴ�ռ�
 *
 * @param layer ���ͷŵ�ȫ���Ӳ�
 */
void free_DenseLayer(DenseLayer* layer)
{
    int n, d, h;
    for (n = 0; n < layer->n_kb; n++)
    {
        for (d = 0; d < layer->kernel_box_dims[0]; d++)
        {
            for (h = 0; h < layer->kernel_box_dims[1]; h++)
            {
                free(layer->kernel_box_group[n][d][h]);
            }
            free(layer->kernel_box_group[n][d]);
        }
        free(layer->kernel_box_group[n]);
    }
    free(layer->kernel_box_group);
    free(layer->bias_array);
    free(layer);
}

/* -------------------------���������Ķ��߳�ʵ��----------------------*/
pthread_mutex_t lock; //�߳���
/**
 * @brief   ������㡾���߳�ʵ�֡�
 *
 * @param input         ��������
 * @param layer         �����
 * @param calc_xx_t     ָ�򼤻���ĺ���ָ��
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @param num_of_thread ���õ��߳���������Ϊ1���߳�ʵ�֣���Ч��Conv()����
 * @return Tensor*
 */
Tensor* Conv_t(Tensor* input, ConvLayer* layer, void* (*calc_activation_t)(void*), int free_input, int num_of_thread)
{
    if (input->dims[0] != layer->kernel_box_dims[0])
    {
        printf("Error���˲㣨 % d�����ں˿����ȼ������������� % d������ȱ���ƥ��\n", layer->kernel_box_dims[0], input->dims[0]);
        exit(EXIT_FAILURE);
    }
    if (num_of_thread < 1)
    {
        printf("Error���߳����費С��1");
        exit(EXIT_FAILURE);
    }

    if (layer->padding == SAME)
    {
        int padding_x, padding_y;
        padding_x = layer->stride_x * (input->dims[2] - 1) - input->dims[2] + layer->kernel_box_dims[2]; // left + right
        padding_y = layer->stride_y * (input->dims[1] - 1) - input->dims[1] + layer->kernel_box_dims[1]; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // ���padding����ʹ'input'ָ��ԭʼ����ĸ��������ͷ�'input'�ǰ�ȫ��
    }

    int output_d = layer->n_kb;
    int output_w, output_h;

    // ��ʽ�е�����ʡ�ԣ���Ϊ��ʱ�����������Ѿ������,��ߴ�Ҳ��Ӧ����
    output_h = ((input->dims[1] /*+ 2*layer->padding */ - layer->kernel_box_dims[1]) / layer->stride_y) + 1;
    output_w = ((input->dims[2] /*+ 2*layer->padding */ - layer->kernel_box_dims[2]) / layer->stride_x) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    // ���̼߳���
    pthread_mutex_init(&lock, NULL); //��ʼ���߳���
    Struct_Conv_T* sct;
    sct = (Struct_Conv_T*)malloc(sizeof(Struct_Conv_T));

    sct->input = input;
    sct->layer = layer;
    sct->output_d = output_d;
    sct->output_h = output_h;
    sct->output_w = output_w;
    sct->output_array = output_array;
    int d = 0, h = 0, w = 0;
    sct->d = &d;
    sct->h = &h;
    sct->w = &w;
    // ��ʼ�������߳�
    pthread_t* tid; // �߳���������ʼ��4��
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // ��������
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_conv_t, sct);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //�ȴ��߳̽���
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("���̼߳������\n");

    if (free_input)
        free_tensor(input);
    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);
    // return activation(output, 1);
    return Activation_t(output, calc_activation_t, 1, num_of_thread);
}

/**
 * @brief   ÿ���̵߳�conv������㺯��
 *
 */
void* calc_conv_t(void* args)
{
    Struct_Conv_T* sct = (Struct_Conv_T*)args;
    int d, h, w, id, by, bx, i, j;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(sct->d);
        h = *(sct->h);
        w = *(sct->w);
        if (d == sct->output_d)
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sct->output_w)
        {
            w = 0;
            h++;
            (*(sct->w)) = 1; // 0++ = 1
            (*(sct->h))++;
            if (h == sct->output_h)
            {
                h = 0;
                d++;
                (*(sct->h)) = 0;
                (*(sct->d))++;
                if (d == sct->output_d)
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sct->w))++;
        pthread_mutex_unlock(&lock);    //����
        sct->output_array[d][h][w] = 0; // ���ڼ�¼����������ÿ����ͨ�����ϵľ��֮��
        for (id = 0; id < sct->input->dims[0]; id++)
        {                                  // �������
            by = h * sct->layer->stride_y; //"begin y" �����ں˴��ڵ��ϱ�Ե��������ϵ�λ��
            bx = w * sct->layer->stride_x; //"begin x" �����ں˴������Ե��������ϵ�λ��
            for (i = 0; i < (sct->layer->kernel_box_dims[1]); i++)
            { // �����ں˴��ڵĸ߶�
                for (j = 0; j < (sct->layer->kernel_box_dims[2]); j++)
                { // �����ں˴��ڵĿ��
                    sct->output_array[d][h][w] += sct->input->T[id][by + i][bx + j] * sct->layer->kernel_box_group[d][id][i][j];
                }
            }
        }
        // ���ƫ��
        sct->output_array[d][h][w] += sct->layer->bias_array[d];
    }
    //printf("\n---------------------------------conv�߳��˳���\n");
    return NULL;
}

/**
 * @brief   ȫ�������㡾���߳�ʵ�֡�
 *
 * @param input         ��������
 * @param layer         �����
 * @param calc_xx_t     ָ�򼤻���ĺ���ָ��
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @param num_of_thread ���õ��߳�����
 * @return Tensor*
 */
Tensor* Dense_t(Tensor* input, DenseLayer* layer, void* (*calc_activation_t)(void*), int free_input, int num_of_thread)
{
    if (input->dims[0] != layer->kernel_box_dims[0] || input->dims[1] != layer->kernel_box_dims[1] || input->dims[2] != layer->kernel_box_dims[2])
    {
        printf("Error������ά�Ȳ�һ��\n");
        printf("���룺��d:%d h:%d w:%d ��| ����ˣ��� d:%d h:%d w:%d��\n", input->dims[0], input->dims[1], input->dims[2], layer->kernel_box_dims[0], layer->kernel_box_dims[1], layer->kernel_box_dims[2]);
        exit(EXIT_FAILURE);
    }

    if (num_of_thread < 1)
    {
        printf("Error���߳����費С��1");
        exit(EXIT_FAILURE);
    }

    int output_d = 1, output_h = 1;
    int output_w = layer->n_kb;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    // ���̼߳���
    pthread_mutex_init(&lock, NULL); //��ʼ���߳���
    Struct_Dense_T* sdt;
    sdt = (Struct_Dense_T*)malloc(sizeof(Struct_Dense_T));
    sdt->input = input;
    sdt->layer = layer;
    sdt->output_w = output_w;
    sdt->output_array = output_array;
    int w = 0;
    sdt->w = &w;
    // ��ʼ�������߳�
    pthread_t* tid; // �߳���������ʼ��4��
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // ��������
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_dense_t, sdt);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //�ȴ��߳̽���
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("���̼߳������\n");

    if (free_input)
        free_tensor(input);
    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);
    // return activation(output, 1);
    return Activation_t(output, calc_activation_t, 1, num_of_thread);
}

/**
 * @brief   ÿ���̵߳�dense������㺯��
 *
 */
void* calc_dense_t(void* args)
{
    Struct_Dense_T* sdt = (Struct_Dense_T*)args;
    int w, id, i, j;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        w = *(sdt->w);
        if (w == sdt->output_w)
        {
            pthread_mutex_unlock(&lock); //����
            break;
        }
        else
            (*(sdt->w))++;
        pthread_mutex_unlock(&lock); //����

        sdt->output_array[0][0][w] = 0;
        for (id = 0; id < sdt->input->dims[0]; id++)
        { // ������ȣ�����ȫ���Ӳ�ͨ��Ϊ1����Ϊ����֮ǰͨ�������չƽ����
            for (i = 0; i < sdt->layer->kernel_box_dims[1]; i++)
            { // �����ں˴��ڵĸ߶�
                for (j = 0; j < sdt->layer->kernel_box_dims[2]; j++)
                { // �����ں˴��ڵĿ��
                    sdt->output_array[0][0][w] += sdt->input->T[id][i][j] * sdt->layer->kernel_box_group[w][id][i][j];
                } // ����by��bx����0������ı䣬��Ϊ�ں�ά�ȵ�������������ά��
            }
        }
        // ���ƫ��
        sdt->output_array[0][0][w] += sdt->layer->bias_array[w];
    }
    //printf("\n---------------------------------dense�߳��˳���\n");
    return NULL;
}

/**
 * @brief   ��������㡾���߳�ʵ�֡�
 *
 * @param input         ��������
 * @param calc_xx_t     ָ����弤����ĺ���ָ��
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @param num_of_thread ���õ��߳�����
 * @return Tensor*
 */
Tensor* Activation_t(Tensor* input, void* (*calc_activation_t)(void*), int free_input, int num_of_thread)
{
    if (num_of_thread < 1)
    { // ���̼߳���
        printf("Error���߳����費С��1");
        exit(EXIT_FAILURE);
    }
    Tensor* output;
    if (free_input)
    {
        output = input;
    }
    else
    {
        float*** output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    // ���̼߳���
    pthread_mutex_init(&lock, NULL); //��ʼ���߳���
    Struct_Activation_T* sat;
    sat = (Struct_Activation_T*)malloc(sizeof(Struct_Activation_T));

    sat->input = input;
    sat->output = output;
    int d = 0, h = 0, w = 0, isAdd = 1;
    float sum = 0;
    sat->d = &d;
    sat->h = &h;
    sat->w = &w;
    sat->sum = &sum;
    sat->isAdd = &isAdd;
    // ��ʼ�������߳�
    pthread_t* tid; // �߳���������ʼ��4��
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // ��������
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_activation_t, sat);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //�ȴ��߳̽���
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("���̼߳������\n");
    return output;
}

/**
 * @brief   ÿ���̵߳�sigmoid������㺯��
 *
 */
void* calc_sigmoid_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sat->output->dims[2])
        {
            w = 0;
            h++;
            (*(sat->w)) = 1; // 0++ = 1
            (*(sat->h))++;
            if (h == sat->output->dims[1])
            {
                h = 0;
                d++;
                (*(sat->h)) = 0;
                (*(sat->d))++;
                if (d == sat->output->dims[0])
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //����
        sat->output->T[d][h][w] = ((float)(1 / (1 + exp((double)-1 * (sat->input->T[d][h][w])))));
    }
    //printf("\n---------------------------------sigmoid�߳��˳���\n");
    return NULL;
}

/**
 * @brief   ÿ���̵߳�softmax������㺯��
 *
 */
void* calc_softmax_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (*(sat->isAdd))
    {                              // �Ӻ�
        pthread_mutex_lock(&lock); //����
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        { // ����߽磬ֱ���˳�
            *(sat->d) = 0;
            *(sat->h) = 0;
            *(sat->w) = 0;
            *(sat->isAdd) = 0;           // �Ӻ����
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sat->output->dims[2])
        {
            w = 0;
            h++;
            (*(sat->w)) = 1; // 0++ = 1
            (*(sat->h))++;
            if (h == sat->output->dims[1])
            {
                h = 0;
                d++;
                (*(sat->h)) = 0;
                (*(sat->d))++;
                if (d == sat->output->dims[0])
                {
                    *(sat->d) = 0;
                    *(sat->h) = 0;
                    *(sat->w) = 0;
                    *(sat->isAdd) = 0;           // �Ӻ����
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock);//����

        float temp = exp(sat->input->T[d][h][w]);
        
        pthread_mutex_lock(&lock); //����
        (*(sat->sum)) += temp;
        pthread_mutex_unlock(&lock); //����
    }
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sat->output->dims[2])
        {
            w = 0;
            h++;
            (*(sat->w)) = 1; // 0++ = 1
            (*(sat->h))++;
            if (h == sat->output->dims[1])
            {
                h = 0;
                d++;
                (*(sat->h)) = 0;
                (*(sat->d))++;
                if (d == sat->output->dims[0])
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //����
        sat->output->T[d][h][w] = exp((sat->input->T[d][h][w])) / *(sat->sum);
    }
    //printf("\n---------------------------------softmax�߳��˳���\n");
    return NULL;
}

/**
 * @brief   ÿ���̵߳�ReLU������㺯��
 *
 */
void* calc_ReLU_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sat->output->dims[2])
        {
            w = 0;
            h++;
            (*(sat->w)) = 1; // 0++ = 1
            (*(sat->h))++;
            if (h == sat->output->dims[1])
            {
                h = 0;
                d++;
                (*(sat->h)) = 0;
                (*(sat->d))++;
                if (d == sat->output->dims[0])
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //����
        sat->output->T[d][h][w] = (sat->input->T[d][h][w] < 0) ? 0 : sat->input->T[d][h][w];
    }
    //printf("\n---------------------------------ReLU�߳��˳���\n");
    return NULL;
}

/**
 * @brief   ÿ���̵߳�ELU������㺯��
 *
 */
void* calc_ELU_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sat->output->dims[2])
        {
            w = 0;
            h++;
            (*(sat->w)) = 1; // 0++ = 1
            (*(sat->h))++;
            if (h == sat->output->dims[1])
            {
                h = 0;
                d++;
                (*(sat->h)) = 0;
                (*(sat->d))++;
                if (d == sat->output->dims[0])
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //����
        sat->output->T[d][h][w] = (sat->input->T[d][h][w] < 0) ? ((float)exp(sat->input->T[d][h][w]) - 1) : sat->input->T[d][h][w];
    }
    //printf("\n---------------------------------ELU�߳��˳���\n");
    return NULL;
}

/**
 * @brief   ÿ���̵߳�linear������㺯��
 *
 */
void* calc_linear_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sat->output->dims[2])
        {
            w = 0;
            h++;
            (*(sat->w)) = 1; // 0++ = 1
            (*(sat->h))++;
            if (h == sat->output->dims[1])
            {
                h = 0;
                d++;
                (*(sat->h)) = 0;
                (*(sat->d))++;
                if (d == sat->output->dims[0])
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //����
        sat->output->T[d][h][w] = sat->input->T[d][h][w];
    }
    //printf("\n---------------------------------linear�߳��˳���\n");
    return NULL;
}

/**
 * @brief   �ػ����㡾���߳�ʵ�֡�
 *
 * @param input         ��������
 * @param calc_xx_t     ָ����弤����ĺ���ָ��
 * @param height        ���ڵĸ߶�
 * @param width         ���ڵĿ��
 * @param stride_x      x����Ĵ��ڲ���
 * @param stride_y      y����Ĵ��ڲ���
 * @param padding       ��䷽ʽ
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @param num_of_thread ���õ��߳�����
 * @return Tensor*
 */
Tensor* Pool_t(Tensor* input, void* (*calc_activation_t)(void*), int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input, int num_of_thread)
{
    if (padding == SAME)
    {
        int padding_x, padding_y;
        padding_x = stride_x * (input->dims[2] - 1) - input->dims[2] + width;  // left + right
        padding_y = stride_y * (input->dims[1] - 1) - input->dims[1] + height; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // ���������ʹ'input'ָ��ԭʼ����ĸ��������ͷ�'input'�ǰ�ȫ��
    }

    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = ((input->dims[2] - width) / stride_x) + 1;
    output_h = ((input->dims[1] - height) / stride_y) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    // ���̼߳���
    pthread_mutex_init(&lock, NULL); //��ʼ���߳���
    Struct_Pool_T* spt;
    spt = (Struct_Pool_T*)malloc(sizeof(Struct_Pool_T));

    spt->input = input;
    spt->output_d = output_d;
    spt->output_h = output_h;
    spt->output_w = output_w;
    spt->stride_x = stride_x;
    spt->stride_y = stride_y;
    spt->height = height;
    spt->width = width;
    spt->output_array = output_array;
    int d = 0, h = 0, w = 0;
    spt->d = &d;
    spt->h = &h;
    spt->w = &w;
    // ��ʼ�������߳�
    pthread_t* tid; // �߳���������ʼ��4��
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // ��������
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_activation_t, spt);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //�ȴ��߳̽���
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("���̼߳������\n");

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);
    return output;
}

/**
 * @brief   ÿ���̵߳�maxpool������㺯��
 *
 */
void* calc_maxpool_t(void* args)
{
    Struct_Pool_T* spt = (Struct_Pool_T*)args;
    int d, h, w, i, j, by, bx;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(spt->d);
        h = *(spt->h);
        w = *(spt->w);
        if (d == spt->output_d)
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == spt->output_w)
        {
            w = 0;
            h++;
            (*(spt->w)) = 1; // 0++ = 1
            (*(spt->h))++;
            if (h == spt->output_h)
            {
                h = 0;
                d++;
                (*(spt->h)) = 0;
                (*(spt->d))++;
                if (d == spt->output_d)
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(spt->w))++;
        pthread_mutex_unlock(&lock); //����

        float max;
        by = h * spt->stride_y;
        bx = w * spt->stride_x;
        max = spt->input->T[d][by][bx];
        for (i = 0; i < spt->height; i++)
        { // �����������ڵĸ߶�
            for (j = 0; j < spt->width; j++)
            { // �����������ڵĿ��
                if ((spt->input->T[d][by + i][bx + j]) > max)
                {
                    max = spt->input->T[d][by + i][bx + j];
                }
            }
        }
        spt->output_array[d][h][w] = max;
    }
    //printf("\n---------------------------------maxpool�߳��˳���\n");
    return NULL;
}

/**
 * @brief   ÿ���̵߳�averagepool������㺯��
 *
 */
void* calc_averagepool_t(void* args)
{
    Struct_Pool_T* spt = (Struct_Pool_T*)args;
    int d, h, w, i, j, by, bx;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(spt->d);
        h = *(spt->h);
        w = *(spt->w);
        if (d == spt->output_d)
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == spt->output_w)
        {
            w = 0;
            h++;
            (*(spt->w)) = 1; // 0++ = 1
            (*(spt->h))++;
            if (h == spt->output_h)
            {
                h = 0;
                d++;
                (*(spt->h)) = 0;
                (*(spt->d))++;
                if (d == spt->output_d)
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(spt->w))++;
        pthread_mutex_unlock(&lock); //����

        float ave;
        by = h * spt->stride_y;
        bx = w * spt->stride_x;
        ave = 0;
        for (i = 0; i < spt->height; i++)
        { // �����������ڵĸ߶�
            for (j = 0; j < spt->width; j++)
            { // �����������ڵĿ��
                ave += spt->input->T[d][by + i][bx + j];
            }
        }
        spt->output_array[d][h][w] = ave / (spt->height * spt->width);
    }
    //printf("\n---------------------------------averagepool�߳��˳���\n");
    return NULL;
}

/**
 * @brief   �ϲ������㡾���߳�ʵ�֡�
 *
 * @param input         ��������
 * @param stride_x      x����Ĵ��ڲ���
 * @param stride_y      y����Ĵ��ڲ���
 * @param padding       ��䷽ʽ
 * @param free_input    �Ƿ��ͷŻ򸲸��������������free_input==1��������������
 * @return Tensor*
 */
Tensor* UpSample_t(Tensor* input, int stride_x, int stride_y, int free_input, int num_of_thread)
{
    int output_d = input->dims[0];
    int output_x = input->dims[1] * stride_x;
    int output_y = input->dims[2] * stride_y;

    float*** output_array = alloc_3D(output_d, output_x, output_y);

    // ���̼߳���
    pthread_mutex_init(&lock, NULL); //��ʼ���߳���
    Struct_UpSample_T* sut;
    sut = (Struct_UpSample_T*)malloc(sizeof(Struct_UpSample_T));

    sut->input = input;
    sut->output_d = output_d;
    sut->output_h = input->dims[1];
    sut->output_w = input->dims[2];
    sut->stride_x = stride_x;
    sut->stride_y = stride_y;
    sut->output_array = output_array;
    int d = 0, h = 0, w = 0;
    sut->d = &d;
    sut->h = &h;
    sut->w = &w;
    // ��ʼ�������߳�
    pthread_t* tid; // �߳���������ʼ��4��
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));

    // ��������
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_upsample_t, sut);
    }

    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //�ȴ��߳̽���
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("���̼߳������\n");

    Tensor* output;
    output = make_tensor(output_d, output_x, output_y, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief   ÿ���̵߳�upsample������㺯��
 *
 */
void* calc_upsample_t(void* args)
{
    Struct_UpSample_T* sut = (Struct_UpSample_T*)args;
    int d, h, w, i, j;
    while (1)
    {
        pthread_mutex_lock(&lock); //����
        d = *(sut->d);
        h = *(sut->h);
        w = *(sut->w);
        if (d == sut->output_d)
        {                                // ����߽磬ֱ���˳�
            pthread_mutex_unlock(&lock); //����
            break;
        }
        if (w == sut->output_w)
        {
            w = 0;
            h++;
            (*(sut->w)) = 1; // 0++ = 1
            (*(sut->h))++;
            if (h == sut->output_h)
            {
                h = 0;
                d++;
                (*(sut->h)) = 0;
                (*(sut->d))++;
                if (d == sut->output_d)
                {
                    pthread_mutex_unlock(&lock); //����
                    break;
                }
            }
        }
        else
            (*(sut->w))++;
        pthread_mutex_unlock(&lock); //����

        for (i = 0; i < sut->stride_x; i++)
        {
            for (j = 0; j < sut->stride_y; j++)
            {
                sut->output_array[d][sut->stride_x * h + i][sut->stride_y * w + j] = sut->input->T[d][h][w];
            }
        }
    }
    //printf("\n---------------------------------upsample�߳��˳���\n");
    return NULL;
}
