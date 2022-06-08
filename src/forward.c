#include "forward.h"

/**
 * @brief   创建不带权重的卷积层，并返回指向该层的指针。
 *
 * @param n_kb      卷积核的数量，这也是偏差的数量
 * @param d_kb      卷积核的深度
 * @param h_kb      卷积核的高度
 * @param w_kb      卷积核的宽度
 * @param stride_x  x方向步长
 * @param stride_y  y方向步长
 * @param padding   填充方式：VALID/SAME
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
 * @brief   创建具有给定权重的卷积层，并返回指向该层的指针
 *
 * @param n_kb          卷积核的数量，这也是偏差的数量
 * @param d_kb          卷积核的深度
 * @param h_kb          卷积核的高度
 * @param w_kb          卷积核的宽度
 * @param weights_array 包含卷积核权重的4D浮点数组（n_kb * d_kb * h_kb * w_kb）
 * @param biases_array  长度为n_kb的浮点偏置数组
 * @param stride_x      x方向步长
 * @param stride_y      y方向步长
 * @param padding       填充方式：VALID/SAME
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
 * @brief   创建没有权重的全连接层，并返回指向该层的指针
 *
 * @param n_kb  卷积核的数量，这也是偏差的数量
 * @param d_kb  卷积核的深度
 * @param h_kb  卷积核的高度
 * @param w_kb  卷积核的宽度
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
 * @brief   创建具有给定权重的全连接层，并返回指向该层的指针
 *
 * @param n_kb          卷积核的数量，这也是偏差的数量
 * @param d_kb          卷积核的深度
 * @param h_kb          卷积核的高度
 * @param w_kb          卷积核的宽度
 * @param weights_array 包含卷积核权重的4D浮点数组（n_kb * d_kb * h_kb * w_kb）
 * @param biases_array  长度为n_kb的浮点偏置数组
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
 * @brief   卷积运算,在应用给定的激活函数之前，通过给定的卷积层获取给定的张量
 *
 * @param input         输入张量
 * @param layer         卷积层
 * @param activation    指向激活函数的函数指针
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* Conv(Tensor* input, ConvLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input)
{
    if (input->dims[0] != layer->kernel_box_dims[0])
    {
        printf("Error：此层（ % d）中内核框的深度及其输入张量（ % d）的深度必须匹配\n", layer->kernel_box_dims[0], input->dims[0]);
        exit(EXIT_FAILURE);
    }

    if (layer->padding == SAME)
    {
        int padding_x, padding_y;
        padding_x = layer->stride_x * (input->dims[2] - 1) - input->dims[2] + layer->kernel_box_dims[2]; // left + right
        padding_y = layer->stride_y * (input->dims[1] - 1) - input->dims[1] + layer->kernel_box_dims[1]; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // 如果padding操作使'input'指向原始输入的副本，则释放'input'是安全的
    }

    int output_d = layer->n_kb;
    int output_w, output_h;

    // 公式中的填充项被省略，因为此时的输入张量已经被填充,其尺寸也相应更新
    output_h = ((input->dims[1] /*+ 2*layer->padding */ - layer->kernel_box_dims[1]) / layer->stride_y) + 1;
    output_w = ((input->dims[2] /*+ 2*layer->padding */ - layer->kernel_box_dims[2]) / layer->stride_x) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, id, by, bx, i, j;

    // 遍历输出数组，逐个计算每个单元格的值
    for (d = 0; d < output_d; d++)
    { // 输出深度
        for (h = 0; h < output_h; h++)
        { // 输出高度
            for (w = 0; w < output_w; w++)
            {                              // 输出宽度
                output_array[d][h][w] = 0; // 用于记录输入张量的每个“通道”上的卷积之和
                for (id = 0; id < input->dims[0]; id++)
                {                             // 输入深度
                    by = h * layer->stride_y; //"begin y" 定义内核窗口的上边缘在输入层上的位置
                    bx = w * layer->stride_x; //"begin x" 定义内核窗口左边缘在输入层上的位置
                    for (i = 0; i < (layer->kernel_box_dims[1]); i++)
                    { // 遍历内核窗口的高度
                        for (j = 0; j < (layer->kernel_box_dims[2]); j++)
                        { // 遍历内核窗口的宽度
                            output_array[d][h][w] += input->T[id][by + i][bx + j] * layer->kernel_box_group[d][id][i][j];
                        }
                    }
                }
                // 添加偏置
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
 * @brief   全连接运算,在应用给定的激活函数之前，通过给定的密集层获取给定的张量
 *
 * @param input         输入张量
 * @param layer         全连接层
 * @param activation    指向激活函数的函数指针
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* Dense(Tensor* input, DenseLayer* layer, Tensor* (*activation)(Tensor*, int), int free_input)
{
    if (input->dims[0] != layer->kernel_box_dims[0] || input->dims[1] != layer->kernel_box_dims[1] || input->dims[2] != layer->kernel_box_dims[2])
    {
        printf("Error：维度不匹配\n");
        printf("输入：（d:%d h:%d w:%d ）| 卷积核：（ d:%d h:%d w:%d）\n", input->dims[0], input->dims[1], input->dims[2], layer->kernel_box_dims[0], layer->kernel_box_dims[1], layer->kernel_box_dims[2]);
        exit(EXIT_FAILURE);
    }

    int output_d = 1, output_h = 1;
    int output_w = layer->n_kb;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, id, i, j;

    // 这个函数遍历输出数组，逐个计算每个单元格的值
    for (d = 0; d < output_d; d++)
    { // 输出深度
        for (h = 0; h < output_h; h++)
        { // 输出高度
            for (w = 0; w < output_w; w++)
            { // 输出宽度
                output_array[d][h][w] = 0;
                for (id = 0; id < input->dims[0]; id++)
                { // 输入深度，对于全连接层通常为1，因为它们之前通常会进行展平操作
                    for (i = 0; i < layer->kernel_box_dims[1]; i++)
                    { // 遍历内核窗口的高度
                        for (j = 0; j < layer->kernel_box_dims[2]; j++)
                        { // 遍历内核窗口的宽度
                            output_array[d][h][w] += input->T[id][i][j] * layer->kernel_box_group[w][id][i][j];
                        } // 这里by和bx都是0，不会改变，因为内核维度等于输入张量层维度
                    }
                }

                // 添加偏置
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
 * @brief sigmoid激活函数
 *
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
 * @brief softmax激活函数
 * y=e^(xi) / ( e^(x1) + e^(x2) + ...  e^(xn) )
 * 
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
 * @brief relu激活函数
 *
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
 * @brief elu激活函数
 *
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
 * @brief 线性激活函数
 *
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
 * @brief   将填充应用于输入张量
 * 如果需要相同的填充(但由于操作参数，对称填充是不可能的)，那么这个函数将遵循tensorflow后端实现，tensor的底部和右侧将获得额外的填充
 * 
 * @param input         输入张量
 * @param padding_x     向左填充+向右填充的总填充像素数量
 * @param padding_y     向上填充+向下填充的总填充像素数量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
        // 上下对称填充
        for (squeeze_along_y = 0; squeeze_along_y < (padding_y / 2); squeeze_along_y++)
        {
            for (x = 0; x < output_w; x++)
            {
                output_array[d][squeeze_along_y][x] = output_array[d][(output_h - 1) - squeeze_along_y][x] = 0;
            }
        }

        // 上下是否对称
        if (padding_y % 2)
        {
            // 填充额外的那个空缺（紧挨着底部填充的那些0）
            for (x = 0; x < output_w; x++)
            {
                output_array[d][(output_h - 1) - (padding_y / 2)][x] = 0;
            }
        }

        // 左右对称填充
        for (squeeze_along_x = 0; squeeze_along_x < (padding_x / 2); squeeze_along_x++)
        {
            for (y = 0; y < output_h; y++)
            {
                output_array[d][y][squeeze_along_x] = output_array[d][y][(output_w - 1) - squeeze_along_x] = 0;
            }
        }

        // 左右是否对称
        if (padding_x % 2)
        {
            // 填充额外的那个空缺（紧挨着右面填充的那些0）
            for (y = 0; y < output_h; y++)
            {
                output_array[d][y][(output_w - 1) - (padding_x / 2)] = 0;
            }
        }

        // 将中间的原数据填入
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
 * @brief   在给定的输入张量上执行最大池化
 *
 * @param input         输入张量
 * @param height        窗口的高度
 * @param width         窗口的宽度
 * @param stride_x      x方向的窗口步长
 * @param stride_y      y方向的窗口步长
 * @param padding       填充方式
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
        free_input = 1; // 如果填充操作使'input'指向原始输入的副本，则释放'input'是安全的
    }

    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = ((input->dims[2] - width) / stride_x) + 1;
    output_h = ((input->dims[1] - height) / stride_y) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, i, j, by, bx;
    float max;

    // 遍历输出数组，逐个计算每个单元格的值
    for (d = 0; d < output_d; d++)
    { // 输出深度
        for (h = 0; h < output_h; h++)
        { // 输出高度
            for (w = 0; w < output_w; w++)
            { //  输出宽度
                by = h * stride_y;
                bx = w * stride_x;
                max = input->T[d][by][bx];
                for (i = 0; i < height; i++)
                { // 遍历整个窗口的高度
                    for (j = 0; j < width; j++)
                    { // 遍历整个窗口的宽度
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
 * @brief    在给定的输入张量上执行均值池化
 *
 * @param input         输入张量
 * @param height        窗口的高度
 * @param width         窗口的宽度
 * @param stride_x      x方向的窗口步长
 * @param stride_y      y方向的窗口步长
 * @param padding       填充方式
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
        free_input = 1; // 如果填充操作使'input'指向原始输入的副本，则释放'input'是安全的
    }

    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = ((input->dims[2] - width) / stride_x) + 1;
    output_h = ((input->dims[1] - height) / stride_y) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, i, j, by, bx;
    float ave;

    // 遍历输出数组，逐个计算每个单元格的值
    for (d = 0; d < output_d; d++)
    { // 输出深度
        for (h = 0; h < output_h; h++)
        { // 输出高度
            for (w = 0; w < output_w; w++)
            { //  输出宽度
                by = h * stride_y;
                bx = w * stride_x;
                ave = 0;
                for (i = 0; i < height; i++)
                { // 遍历整个窗口的高度
                    for (j = 0; j < width; j++)
                    { // 遍历整个窗口的宽度
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
 * @brief 在给定的输入张量上执行上采样
 * (d,x,y)->(d,m*x,n*y)
 *
 * @param input         输入张量
 * @param stride_x      x方向的上采样倍率
 * @param stride_y      y方向的上采样倍率
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* UpSample(Tensor* input, int stride_x, int stride_y, int free_input)
{
    int output_d = input->dims[0];
    int output_x = input->dims[1] * stride_x;
    int output_y = input->dims[2] * stride_y;
    float*** output_array = alloc_3D(output_d, output_x, output_y);

    int d, x, y, i, j;
    // 遍历输出数组，逐个计算每个单元格的值
    for (d = 0; d < output_d; d++) // 输出深度
    {
        for (x = 0; x < input->dims[1]; x++) // 输出高度
        {
            for (y = 0; y < input->dims[2]; y++) // 输出高度
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
 * @brief 在给定的输入张量上执行拼接
 * (d1, x, y)+(d2, x, y) -> (d1+d2, x, y)
 *
 * @param input1 输入张量1
 * @param input2 输入张量2
 * @return Tensor*
 */
Tensor* Concatenate(Tensor* input1, Tensor* input2, int free_input)
{
    if (input1->dims[1] != input2->dims[1] || input1->dims[2] != input2->dims[2])
    {
        printf("Error: 两张量h、w不一致，无法拼接。\n");
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
 * @brief 在给定的输入张量上执行批标准化
 * y = gamma(x-moving_mean)/sqrt(moving_variance+ε)+beta
 * 
 * @param input             输入张量
 * @param gamma             缩放系数
 * @param beta              偏移系数
 * @param moving_mean       均值系数
 * @param moving_variance   方差系数
 * @param free_input        是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* BatchNormalization(Tensor* input, float** gamma, float** beta, float** moving_mean, float** moving_variance, int free_input)
{
    int output_d = input->dims[0];
    int output_h = input->dims[1];
    int output_w = input->dims[2];
    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w;
    // 遍历输出数组，逐个计算每个单元格的值

    for (d = 0; d < output_d; d++) // 输出长度
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
 * @brief   将输入张量展平为其宽度，以便输出深度和高度为1：(d,h,w) -> (1,1,d*h*w)
 *
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* FlattenW(Tensor* input, int free_input)
{
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_d = 1, output_h = 1;
    int output_w = input_d * input_h * input_w;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    int d, h, w, i = 0;

    // keras中shape为（h,w,d）,本库定义格式为（d,h,w），keras默认为（h,w,d），不能直接摊平
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
 * @brief   将输入张量展平到其高度，使输出深度和宽度为1：(d,h,w) -> (1,d*h*w,1)
 *
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
 * @brief   将输入张量展平到其深度，使输出高度和宽度为1：(d,h,w) -> (d*h*w,1,1)
 *
 * @param input         输入张量
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
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
 * @brief   对数组中的张量进行逐元素求和
 *
 * @param input1        张量1
 * @param input2        张量2
 * @param free_inputs   是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* Add(Tensor* input1, Tensor* input2, int free_inputs)
{

    if (input1->dims[0] != input2->dims[0] || input1->dims[1] != input2->dims[1] || input1->dims[2] != input2->dims[2])
    {
        printf("Error: 两张量d、h、w不一致，无法相加。\n");
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
 * @brief   对张量沿着某维度进行元素求和后，除以该维度的长度n，得到元素均值
 *
 * @param input         张量
 * @param nD            均值池化的维度，1或2
 * @param free_inputs   是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* GlobalAveragePooling(Tensor* input, int nD, int free_inputs)
{
    if (nD != 1 && nD != 2)
    {
        printf("Error：dims应为1或2\n");
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
 * @brief   用于打印张量到控制台
 *
 * @param t 待打印的张量指针
 */
void print_tensor(Tensor* t)
{

    printf("维度：%d,%d,%d\n\n---------------------------------------\n", t->dims[0], t->dims[1], t->dims[2]);

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
 * @brief   为具有维度（b*d*h*w）的4D浮点数组分配内存并返回指针。
 *
 * @param b     维度 0, size of float *** array
 * @param d     维度 1, size of float ** array
 * @param h     维度 2, size of float * array
 * @param w     维度 3, size of float array
 * @return float****
 */
float**** alloc_4D(int b, int d, int h, int w)
{
    float**** newArr;
    newArr = malloc(b * sizeof(float***));
    if (newArr == NULL)
    {
        printf("Error: 无法为alloc_4D.newArr分配内存\n");
        exit(EXIT_FAILURE);
    }

    int i, j, k;
    for (i = 0; i < b; i++)
    {
        newArr[i] = malloc(d * sizeof(float**));
        if (newArr[i] == NULL)
        {
            printf("Error: 无法为alloc_4D.newArr[%d]分配内存\n", i);
            exit(EXIT_FAILURE);
        }
        for (j = 0; j < d; j++)
        {
            newArr[i][j] = malloc(h * sizeof(float*));
            if (newArr[i][j] == NULL)
            {
                printf("Error: 无法为alloc_4D.new[%d][%d]分配内存\n", i, j);
                exit(EXIT_FAILURE);
            }
            for (k = 0; k < h; k++)
            {
                newArr[i][j][k] = malloc(w * sizeof(float));
                if (newArr[i][j][k] == NULL)
                {
                    printf("Error: 无法为alloc_4D.new[%d][%d][%d]分配内存\n", i, j, k);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    return newArr;
}

/**
 * @brief   为具有维度（d*h*w）的3D浮点数组分配内存并返回指针。
 *
 * @param d     维度 0, size of float ** array
 * @param h     维度 1, size of float * array
 * @param w     维度 2, size of float array
 * @return float***
 */
float*** alloc_3D(int d, int h, int w)
{
    float*** newArr;
    newArr = malloc(d * sizeof(float**));
    if (newArr == NULL)
    {
        printf("Error: 无法为alloc_3D.newArr分配内存\n");
        exit(EXIT_FAILURE);
    }

    int i, j;
    for (i = 0; i < d; i++)
    {
        newArr[i] = malloc(h * sizeof(float*));
        if (newArr[i] == NULL)
        {
            printf("Error: 无法为alloc_3D.newArr[%d]分配内存\n", i);
            exit(EXIT_FAILURE);
        }
        for (j = 0; j < h; j++)
        {
            newArr[i][j] = malloc(w * sizeof(float));
            if (newArr[i][j] == NULL)
            {
                printf("Error: 无法为alloc_3D.newArr[%d][%d]分配内存\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    return newArr;
}

/**
 * @brief 打印有关给定卷积层的信息
 *
 * @param layer 待打印的卷积层
 */
void print_conv_details(ConvLayer layer)
{
    printf("卷积层位于：%p\n", &layer);
    printf("\t卷积核数量：%d\n", layer.n_kb);
    printf("\t卷积核维度：%d,%d,%d\n", layer.kernel_box_dims[0], layer.kernel_box_dims[1], layer.kernel_box_dims[2]);
    printf("\tx步长：%d\n", layer.stride_x);
    printf("\ty步长：%d\n", layer.stride_y);
    printf("\t填充方式：%d\n\n", layer.padding);

    int n, d, h, w;
    printf("\t权重信息：\n");
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
    printf("\t偏置信息：\n");
    for (n = 0; n < layer.n_kb; n++)
    {
        printf("\t\t%f\n", layer.bias_array[n]);
    }
}

/**
 * @brief 打印有关给定全连接层的信息
 * 
 * @param layer 待打印的卷积层
 */
void print_dense_details(DenseLayer layer)
{
    printf("卷积层位于：%p\n", &layer);
    printf("\t卷积核数量：%d\n", layer.n_kb);
    printf("\t卷积核维度：%d,%d,%d\n", layer.kernel_box_dims[0], layer.kernel_box_dims[1], layer.kernel_box_dims[2]);

    int n, d, h, w;
    printf("\t权重信息：\n");
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
    printf("\t偏置信息：\n");
    for (n = 0; n < layer.n_kb; n++)
    {
        printf("\t\t%f\n", layer.bias_array[n]);
    }
}

/**
 * @brief 释放张量t占用的内存
 *
 * @param t 待释放内存的张量
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
 * @brief       创建并配置新的（d*h*w）维度张量
 *
 * @param d     张量深度
 * @param h     张量高度
 * @param w     张量宽度
 * @param array 具有维度（d*h*w）的3D浮点数组，将从中构建张量
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
 * @brief 释放分配给给定卷积层的内存空间
 *
 * @param layer 待释放的卷积层
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
 * @brief 释放分配给给定全连接层的内存空间
 *
 * @param layer 待释放的全连接层
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

/* -------------------------上述函数的多线程实现----------------------*/
pthread_mutex_t lock; //线程锁
/**
 * @brief   卷积运算【多线程实现】
 *
 * @param input         输入张量
 * @param layer         卷积层
 * @param calc_xx_t     指向激活函数的函数指针
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @param num_of_thread 调用的线程数量，若为1则单线程实现，等效于Conv()函数
 * @return Tensor*
 */
Tensor* Conv_t(Tensor* input, ConvLayer* layer, void* (*calc_activation_t)(void*), int free_input, int num_of_thread)
{
    if (input->dims[0] != layer->kernel_box_dims[0])
    {
        printf("Error：此层（ % d）中内核框的深度及其输入张量（ % d）的深度必须匹配\n", layer->kernel_box_dims[0], input->dims[0]);
        exit(EXIT_FAILURE);
    }
    if (num_of_thread < 1)
    {
        printf("Error：线程数需不小于1");
        exit(EXIT_FAILURE);
    }

    if (layer->padding == SAME)
    {
        int padding_x, padding_y;
        padding_x = layer->stride_x * (input->dims[2] - 1) - input->dims[2] + layer->kernel_box_dims[2]; // left + right
        padding_y = layer->stride_y * (input->dims[1] - 1) - input->dims[1] + layer->kernel_box_dims[1]; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // 如果padding操作使'input'指向原始输入的副本，则释放'input'是安全的
    }

    int output_d = layer->n_kb;
    int output_w, output_h;

    // 公式中的填充项被省略，因为此时的输入张量已经被填充,其尺寸也相应更新
    output_h = ((input->dims[1] /*+ 2*layer->padding */ - layer->kernel_box_dims[1]) / layer->stride_y) + 1;
    output_w = ((input->dims[2] /*+ 2*layer->padding */ - layer->kernel_box_dims[2]) / layer->stride_x) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    // 多线程计算
    pthread_mutex_init(&lock, NULL); //初始化线程锁
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
    // 初始化所有线程
    pthread_t* tid; // 线程数量，初始化4个
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // 分配任务
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_conv_t, sct);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //等待线程结束
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("多线程计算完成\n");

    if (free_input)
        free_tensor(input);
    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);
    // return activation(output, 1);
    return Activation_t(output, calc_activation_t, 1, num_of_thread);
}

/**
 * @brief   每个线程的conv具体计算函数
 *
 */
void* calc_conv_t(void* args)
{
    Struct_Conv_T* sct = (Struct_Conv_T*)args;
    int d, h, w, id, by, bx, i, j;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(sct->d);
        h = *(sct->h);
        w = *(sct->w);
        if (d == sct->output_d)
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sct->w))++;
        pthread_mutex_unlock(&lock);    //解锁
        sct->output_array[d][h][w] = 0; // 用于记录输入张量的每个“通道”上的卷积之和
        for (id = 0; id < sct->input->dims[0]; id++)
        {                                  // 输入深度
            by = h * sct->layer->stride_y; //"begin y" 定义内核窗口的上边缘在输入层上的位置
            bx = w * sct->layer->stride_x; //"begin x" 定义内核窗口左边缘在输入层上的位置
            for (i = 0; i < (sct->layer->kernel_box_dims[1]); i++)
            { // 遍历内核窗口的高度
                for (j = 0; j < (sct->layer->kernel_box_dims[2]); j++)
                { // 遍历内核窗口的宽度
                    sct->output_array[d][h][w] += sct->input->T[id][by + i][bx + j] * sct->layer->kernel_box_group[d][id][i][j];
                }
            }
        }
        // 添加偏置
        sct->output_array[d][h][w] += sct->layer->bias_array[d];
    }
    //printf("\n---------------------------------conv线程退出。\n");
    return NULL;
}

/**
 * @brief   全连接运算【多线程实现】
 *
 * @param input         输入张量
 * @param layer         卷积层
 * @param calc_xx_t     指向激活函数的函数指针
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @param num_of_thread 调用的线程数量
 * @return Tensor*
 */
Tensor* Dense_t(Tensor* input, DenseLayer* layer, void* (*calc_activation_t)(void*), int free_input, int num_of_thread)
{
    if (input->dims[0] != layer->kernel_box_dims[0] || input->dims[1] != layer->kernel_box_dims[1] || input->dims[2] != layer->kernel_box_dims[2])
    {
        printf("Error：输入维度不一致\n");
        printf("输入：（d:%d h:%d w:%d ）| 卷积核：（ d:%d h:%d w:%d）\n", input->dims[0], input->dims[1], input->dims[2], layer->kernel_box_dims[0], layer->kernel_box_dims[1], layer->kernel_box_dims[2]);
        exit(EXIT_FAILURE);
    }

    if (num_of_thread < 1)
    {
        printf("Error：线程数需不小于1");
        exit(EXIT_FAILURE);
    }

    int output_d = 1, output_h = 1;
    int output_w = layer->n_kb;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    // 多线程计算
    pthread_mutex_init(&lock, NULL); //初始化线程锁
    Struct_Dense_T* sdt;
    sdt = (Struct_Dense_T*)malloc(sizeof(Struct_Dense_T));
    sdt->input = input;
    sdt->layer = layer;
    sdt->output_w = output_w;
    sdt->output_array = output_array;
    int w = 0;
    sdt->w = &w;
    // 初始化所有线程
    pthread_t* tid; // 线程数量，初始化4个
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // 分配任务
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_dense_t, sdt);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //等待线程结束
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("多线程计算完成\n");

    if (free_input)
        free_tensor(input);
    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);
    // return activation(output, 1);
    return Activation_t(output, calc_activation_t, 1, num_of_thread);
}

/**
 * @brief   每个线程的dense具体计算函数
 *
 */
void* calc_dense_t(void* args)
{
    Struct_Dense_T* sdt = (Struct_Dense_T*)args;
    int w, id, i, j;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        w = *(sdt->w);
        if (w == sdt->output_w)
        {
            pthread_mutex_unlock(&lock); //解锁
            break;
        }
        else
            (*(sdt->w))++;
        pthread_mutex_unlock(&lock); //解锁

        sdt->output_array[0][0][w] = 0;
        for (id = 0; id < sdt->input->dims[0]; id++)
        { // 输入深度，对于全连接层通常为1，因为它们之前通常会进行展平操作
            for (i = 0; i < sdt->layer->kernel_box_dims[1]; i++)
            { // 遍历内核窗口的高度
                for (j = 0; j < sdt->layer->kernel_box_dims[2]; j++)
                { // 遍历内核窗口的宽度
                    sdt->output_array[0][0][w] += sdt->input->T[id][i][j] * sdt->layer->kernel_box_group[w][id][i][j];
                } // 这里by和bx都是0，不会改变，因为内核维度等于输入张量层维度
            }
        }
        // 添加偏置
        sdt->output_array[0][0][w] += sdt->layer->bias_array[w];
    }
    //printf("\n---------------------------------dense线程退出。\n");
    return NULL;
}

/**
 * @brief   激活函数运算【多线程实现】
 *
 * @param input         输入张量
 * @param calc_xx_t     指向具体激活函数的函数指针
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @param num_of_thread 调用的线程数量
 * @return Tensor*
 */
Tensor* Activation_t(Tensor* input, void* (*calc_activation_t)(void*), int free_input, int num_of_thread)
{
    if (num_of_thread < 1)
    { // 单线程计算
        printf("Error：线程数需不小于1");
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

    // 多线程计算
    pthread_mutex_init(&lock, NULL); //初始化线程锁
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
    // 初始化所有线程
    pthread_t* tid; // 线程数量，初始化4个
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // 分配任务
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_activation_t, sat);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //等待线程结束
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("多线程计算完成\n");
    return output;
}

/**
 * @brief   每个线程的sigmoid具体计算函数
 *
 */
void* calc_sigmoid_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //解锁
        sat->output->T[d][h][w] = ((float)(1 / (1 + exp((double)-1 * (sat->input->T[d][h][w])))));
    }
    //printf("\n---------------------------------sigmoid线程退出。\n");
    return NULL;
}

/**
 * @brief   每个线程的softmax具体计算函数
 *
 */
void* calc_softmax_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (*(sat->isAdd))
    {                              // 加和
        pthread_mutex_lock(&lock); //上锁
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        { // 到达边界，直接退出
            *(sat->d) = 0;
            *(sat->h) = 0;
            *(sat->w) = 0;
            *(sat->isAdd) = 0;           // 加和完成
            pthread_mutex_unlock(&lock); //解锁
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
                    *(sat->isAdd) = 0;           // 加和完成
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock);//解锁

        float temp = exp(sat->input->T[d][h][w]);
        
        pthread_mutex_lock(&lock); //上锁
        (*(sat->sum)) += temp;
        pthread_mutex_unlock(&lock); //解锁
    }
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //解锁
        sat->output->T[d][h][w] = exp((sat->input->T[d][h][w])) / *(sat->sum);
    }
    //printf("\n---------------------------------softmax线程退出。\n");
    return NULL;
}

/**
 * @brief   每个线程的ReLU具体计算函数
 *
 */
void* calc_ReLU_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //解锁
        sat->output->T[d][h][w] = (sat->input->T[d][h][w] < 0) ? 0 : sat->input->T[d][h][w];
    }
    //printf("\n---------------------------------ReLU线程退出。\n");
    return NULL;
}

/**
 * @brief   每个线程的ELU具体计算函数
 *
 */
void* calc_ELU_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //解锁
        sat->output->T[d][h][w] = (sat->input->T[d][h][w] < 0) ? ((float)exp(sat->input->T[d][h][w]) - 1) : sat->input->T[d][h][w];
    }
    //printf("\n---------------------------------ELU线程退出。\n");
    return NULL;
}

/**
 * @brief   每个线程的linear具体计算函数
 *
 */
void* calc_linear_t(void* args)
{
    Struct_Activation_T* sat = (Struct_Activation_T*)args;
    int d, h, w;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(sat->d);
        h = *(sat->h);
        w = *(sat->w);
        if (d == sat->output->dims[0])
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sat->w))++;
        pthread_mutex_unlock(&lock); //解锁
        sat->output->T[d][h][w] = sat->input->T[d][h][w];
    }
    //printf("\n---------------------------------linear线程退出。\n");
    return NULL;
}

/**
 * @brief   池化运算【多线程实现】
 *
 * @param input         输入张量
 * @param calc_xx_t     指向具体激活函数的函数指针
 * @param height        窗口的高度
 * @param width         窗口的宽度
 * @param stride_x      x方向的窗口步长
 * @param stride_y      y方向的窗口步长
 * @param padding       填充方式
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @param num_of_thread 调用的线程数量
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
        free_input = 1; // 如果填充操作使'input'指向原始输入的副本，则释放'input'是安全的
    }

    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = ((input->dims[2] - width) / stride_x) + 1;
    output_h = ((input->dims[1] - height) / stride_y) + 1;

    float*** output_array = alloc_3D(output_d, output_h, output_w);

    // 多线程计算
    pthread_mutex_init(&lock, NULL); //初始化线程锁
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
    // 初始化所有线程
    pthread_t* tid; // 线程数量，初始化4个
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));
    // 分配任务
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_activation_t, spt);
    }
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //等待线程结束
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("多线程计算完成\n");

    Tensor* output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if (free_input)
        free_tensor(input);
    return output;
}

/**
 * @brief   每个线程的maxpool具体计算函数
 *
 */
void* calc_maxpool_t(void* args)
{
    Struct_Pool_T* spt = (Struct_Pool_T*)args;
    int d, h, w, i, j, by, bx;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(spt->d);
        h = *(spt->h);
        w = *(spt->w);
        if (d == spt->output_d)
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(spt->w))++;
        pthread_mutex_unlock(&lock); //解锁

        float max;
        by = h * spt->stride_y;
        bx = w * spt->stride_x;
        max = spt->input->T[d][by][bx];
        for (i = 0; i < spt->height; i++)
        { // 遍历整个窗口的高度
            for (j = 0; j < spt->width; j++)
            { // 遍历整个窗口的宽度
                if ((spt->input->T[d][by + i][bx + j]) > max)
                {
                    max = spt->input->T[d][by + i][bx + j];
                }
            }
        }
        spt->output_array[d][h][w] = max;
    }
    //printf("\n---------------------------------maxpool线程退出。\n");
    return NULL;
}

/**
 * @brief   每个线程的averagepool具体计算函数
 *
 */
void* calc_averagepool_t(void* args)
{
    Struct_Pool_T* spt = (Struct_Pool_T*)args;
    int d, h, w, i, j, by, bx;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(spt->d);
        h = *(spt->h);
        w = *(spt->w);
        if (d == spt->output_d)
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(spt->w))++;
        pthread_mutex_unlock(&lock); //解锁

        float ave;
        by = h * spt->stride_y;
        bx = w * spt->stride_x;
        ave = 0;
        for (i = 0; i < spt->height; i++)
        { // 遍历整个窗口的高度
            for (j = 0; j < spt->width; j++)
            { // 遍历整个窗口的宽度
                ave += spt->input->T[d][by + i][bx + j];
            }
        }
        spt->output_array[d][h][w] = ave / (spt->height * spt->width);
    }
    //printf("\n---------------------------------averagepool线程退出。\n");
    return NULL;
}

/**
 * @brief   上采样运算【多线程实现】
 *
 * @param input         输入张量
 * @param stride_x      x方向的窗口步长
 * @param stride_y      y方向的窗口步长
 * @param padding       填充方式
 * @param free_input    是否释放或覆盖输入张量，如果free_input==1，则丢弃输入张量
 * @return Tensor*
 */
Tensor* UpSample_t(Tensor* input, int stride_x, int stride_y, int free_input, int num_of_thread)
{
    int output_d = input->dims[0];
    int output_x = input->dims[1] * stride_x;
    int output_y = input->dims[2] * stride_y;

    float*** output_array = alloc_3D(output_d, output_x, output_y);

    // 多线程计算
    pthread_mutex_init(&lock, NULL); //初始化线程锁
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
    // 初始化所有线程
    pthread_t* tid; // 线程数量，初始化4个
    tid = (pthread_t*)malloc(sizeof(pthread_t) * 4);
    if (num_of_thread > 4)
        tid = (pthread_t*)realloc(tid, num_of_thread * sizeof(pthread_t));

    // 分配任务
    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_create(&tid[i], NULL, calc_upsample_t, sut);
    }

    for (int i = 0; i < num_of_thread; i++)
    {
        pthread_join(tid[i], NULL); //等待线程结束
    }
    free(tid);
    pthread_mutex_destroy(&lock);
    //printf("多线程计算完成\n");

    Tensor* output;
    output = make_tensor(output_d, output_x, output_y, output_array);

    if (free_input)
        free_tensor(input);

    return output;
}

/**
 * @brief   每个线程的upsample具体计算函数
 *
 */
void* calc_upsample_t(void* args)
{
    Struct_UpSample_T* sut = (Struct_UpSample_T*)args;
    int d, h, w, i, j;
    while (1)
    {
        pthread_mutex_lock(&lock); //上锁
        d = *(sut->d);
        h = *(sut->h);
        w = *(sut->w);
        if (d == sut->output_d)
        {                                // 到达边界，直接退出
            pthread_mutex_unlock(&lock); //解锁
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
                    pthread_mutex_unlock(&lock); //解锁
                    break;
                }
            }
        }
        else
            (*(sut->w))++;
        pthread_mutex_unlock(&lock); //解锁

        for (i = 0; i < sut->stride_x; i++)
        {
            for (j = 0; j < sut->stride_y; j++)
            {
                sut->output_array[d][sut->stride_x * h + i][sut->stride_y * w + j] = sut->input->T[d][h][w];
            }
        }
    }
    //printf("\n---------------------------------upsample线程退出。\n");
    return NULL;
}
