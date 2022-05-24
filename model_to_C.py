def extract_all(model):
    # 配置精度
    D_or_F = 'float' # 'double' or 'float'
    jing_du = ".7f"
    
    if D_or_F == 'float': # 选'f'会被gcc识别为float型
        d_or_f = 'f'
    else :  # 选''会被gcc识别为double型
        d_or_f = ''
    firstLayerConfig = model.get_layer(index=0).get_config()
    fh = open("model.h", "w+")
    fc = open("model.c", "w+")
    fh.write('#ifndef MODEL\n#define MODEL\n\n\n')
    fc.write('#include "forward.h"\n#include "model.h"\n')
    fc.write('\nint main(int argc, char *argv[]){\n\n')
    fc.write('\n\n\t/******************  初始化输入张量  ******************/\n')
    fc.write('\t'+D_or_F+'*** singal_array;\n')
    
    if len(firstLayerConfig['batch_input_shape']) == 4:  #(none,h,w,d)
        fh.write(D_or_F+' singal[{2}][{0}][{1}] = {{ 0 }};\n'.format(firstLayerConfig['batch_input_shape'][1], firstLayerConfig['batch_input_shape'][2], firstLayerConfig['batch_input_shape'][3]))
        fc.write('\tsingal_array = alloc_3D({2}, {0}, {1});\n'.format(firstLayerConfig['batch_input_shape'][1], firstLayerConfig['batch_input_shape'][2], firstLayerConfig['batch_input_shape'][3]))
        fc.write('\tfor (int i = 0; i < {0}; i++) {{\n'.format(firstLayerConfig['batch_input_shape'][3]))
        fc.write('\t\tfor (int j = 0; j < {0}; j++) {{\n'.format(firstLayerConfig['batch_input_shape'][1]))
        fc.write('\t\t\tsingal_array[i][j] = singal[i][j];\n\t\t}\n\t}\n')       
        fc.write('\tTensor* {0};\n'.format((model.get_layer(index=0).input.name).split('/')[0]))
        fc.write('\t{0} = make_tensor({3}, {1}, {2}, singal_array);\n'.format((model.get_layer(index=0).input.name).split('/')[0], firstLayerConfig['batch_input_shape'][1], firstLayerConfig['batch_input_shape'][2], firstLayerConfig['batch_input_shape'][3]))
    elif len(firstLayerConfig['batch_input_shape']) == 3:  #(none,w,d)
        fh.write(D_or_F+' singal[{1}][{0}] = {{ 0 }};\n'.format(firstLayerConfig['batch_input_shape'][1], firstLayerConfig['batch_input_shape'][2]))
        fc.write('\tsingal_array = alloc_3D({1}, 1, {0});\n'.format(firstLayerConfig['batch_input_shape'][1], firstLayerConfig['batch_input_shape'][2]))
        fc.write('\tfor (int i = 0; i < {0}; i++) {{\n'.format(firstLayerConfig['batch_input_shape'][2]))
        fc.write('\t\tsingal_array[i][0] = singal[i];\n\t}\n')         
        fc.write('\tTensor* {0};\n'.format((model.get_layer(index=0).input.name).split('/')[0]))
        fc.write('\t{2} = make_tensor({1}, 1, {0}, singal_array);\n'.format(firstLayerConfig['batch_input_shape'][1], firstLayerConfig['batch_input_shape'][2], (model.get_layer(index=0).input.name).split('/')[0]))
    elif len(firstLayerConfig['batch_input_shape']) == 2: #(none,w,)
        fh.write(D_or_F+' singal[{0}] = {{ 0 }};\n'.format(firstLayerConfig['batch_input_shape'][1]))
        fc.write('\tsingal_array = alloc_3D(1, 1, {0});\n'.format(firstLayerConfig['batch_input_shape'][1]))
        fc.write('\tfor (int i = 0; i < {0}; i++) {{\n\t\tsingal_array[0][0][i] = singal[i];\n\t}}\n'.format(firstLayerConfig['batch_input_shape'][1]))
        fc.write('\tTensor* {0};\n'.format((model.get_layer(index=0).input.name).split('/')[0]))
        fc.write('\t{1} = make_tensor(1, 1, {0}, singal_array);\n'.format(firstLayerConfig['batch_input_shape'][1], (model.get_layer(index=0).input.name).split('/')[0]))
    else:
        print('【初始化输入张量维度错误】')
    fc.write('\n\n\t/******************  创建各层   ******************/\n')
    for layer in model.layers:
        # 卷积层~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            weights = np.transpose(layer.get_weights()[0])  # n,d,w,h
            w_len = len(weights)
            # 偏置===========================================
            if len(layer.get_weights())==2:  # 含偏置
                biases = layer.get_weights()[1]
                b_len = len(biases)
                fh.write(D_or_F+" {0}_biases[{1}]=\n{{".format(layer.name, b_len))
                for n in range(b_len):
                    fh.write(format(biases[n], jing_du) + d_or_f+ ",")
                fh.write('};\n')
            elif len(layer.get_weights())==1:  # 不含偏置
                fh.write(D_or_F+" {0}_biases[{1}]=\n{{".format(layer.name, w_len))
                fh.write('0};\n')
            # 权重===========================================
            fh.write(D_or_F+" {0}_weights[{1}][{2}][{3}][{4}]=\n{{".format(layer.name, w_len, layer.input_shape[3], layer.kernel_size[0], layer.kernel_size[1]))
            for n in range(w_len):
                fh.write("{")
                for d in range(layer.input_shape[3]):
                    fh.write("{")
                    for h in range(layer.kernel_size[0]):
                        fh.write("{")
                        for w in range(layer.kernel_size[1]):
                            fh.write(format(weights[n][d][w][h], jing_du) + d_or_f+",")
                        fh.write("},\n")    
                    fh.write("},\n")
                fh.write("},\n")
            fh.write("};\n")
            fc.write('\n\t// ---------- 创建一个卷积层 ----------\n')
            # 生成 float ***weights_array
            fc.write('\t// 生成 float ****weights_array\n')
            fc.write('\t'+D_or_F+' ****{0}_pw;\n'.format(layer.name))
            fc.write('\t{0}_pw = alloc_4D({1}, {2}, {3}, {4});\n'.format(layer.name, w_len, layer.input_shape[3], layer.kernel_size[0], layer.kernel_size[1]))
            fc.write('\tfor(int i = 0; i < {0}; i++){{\n'.format(w_len))
            fc.write('\t\tfor(int j = 0; j < {0}; j++){{\n'.format(layer.input_shape[3]))
            fc.write('\t\t\tfor(int k = 0; k < {0}; k++){{\n'.format(layer.kernel_size[0]))
            fc.write('\t\t\t{0}_pw[i][j][k] = {0}_weights[i][j][k];\n'.format(layer.name))
            fc.write('\t\t\t}\n')
            fc.write('\t\t}\n')
            fc.write('\t}\n')
            # 创建一个卷积层
            fc.write('\tConvLayer *_{0};\n\t_{0} = new_Conv({1}, {2}, {3}, {4}, {0}_pw, &{0}_biases, {5}, {6}, {7});\n'.format(layer.name, w_len, layer.input_shape[3], layer.kernel_size[0], layer.kernel_size[1], layer.strides[0], layer.strides[1], layer.padding.upper()))
        elif isinstance(layer, keras.layers.convolutional.Conv1D):
            # 偏置===========================================
            biases = layer.get_weights()[1]
            b_len = len(biases)
            fh.write(D_or_F+" {0}_biases[{1}]=\n{{".format(layer.name, b_len))
            for n in range(b_len):
                fh.write(format(biases[n], jing_du) + d_or_f+ ",")
            fh.write('};\n')
            # 权重===========================================
            weights = np.transpose(layer.get_weights()[0])
            w_len = len(weights)
            fh.write(D_or_F+" {0}_weights[{1}][{2}][{3}]=\n{{".format(layer.name, w_len, layer.input_shape[2], layer.kernel_size[0]))
            for n in range(w_len):
                fh.write("{")
                for d in range(layer.input_shape[2]):
                    fh.write("{")
                    for l in range(layer.kernel_size[0]):
                        fh.write(format(weights[n][d][l], jing_du) + d_or_f+",")
                    fh.write("},\n")
                fh.write("},\n")
            fh.write("};\n")
            fc.write('\n\t// ---------- 创建一个卷积层 ----------\n')
            # 生成 float ***weights_array
            fc.write('\t// 生成 float ****weights_array\n')
            fc.write('\t'+D_or_F+' ****{0}_pw;\n'.format(layer.name))
            fc.write('\t{0}_pw = alloc_4D({1}, {2}, 1, {3});\n'.format(layer.name, layer.filters, layer.input_shape[2], layer.kernel_size[0]))
            fc.write('\tfor(int i = 0; i < {0}; i++){{\n'.format(layer.filters))
            fc.write('\t\tfor(int j = 0; j < {0}; j++){{\n'.format(layer.input_shape[2]))
            fc.write('\t\t\t{0}_pw[i][j][0] = {0}_weights[i][j];\n'.format(layer.name))
            fc.write('\t\t}\n')
            fc.write('\t}\n')
            # 创建一个卷积层
            fc.write('\tConvLayer *_{0};\n\t_{0} = new_Conv({1}, {2}, 1, {3}, {0}_pw, &{0}_biases, 1, {4}, {5});\n'.format(layer.name, layer.filters, layer.input_shape[2], layer.kernel_size[0], layer.strides[0], layer.padding.upper()))
        # 全连接层~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif isinstance(layer, keras.layers.core.Dense):
            # 偏置===========================================
            biases = layer.get_weights()[1]
            b_len = len(biases)
            fh.write(D_or_F+" {0}_biases[{1}]={{".format(layer.name, b_len))
            for n in range(b_len):
                fh.write(format(biases[n], jing_du) + d_or_f+",")
            fh.write('};\n')
            # 权重===========================================
            weights = np.transpose(layer.get_weights()[0])#(784,32) -> (32,784)
            w_len = len(weights)
            fh.write(D_or_F+" {0}_weights[{1}][{2}]={{".format(layer.name, w_len, layer.input_shape[1]))
            for n in range(w_len): #32
                fh.write("{")
                for d in range(layer.input_shape[1]): #784
                    fh.write(format(weights[n][d], jing_du) +d_or_f+ ",")
                fh.write("},\n")
            fh.write("};\n")
            fc.write('\n\t// ---------- 创建一个全连接层 ----------\n')
            # 生成 float ***weights_array
            fc.write('\t// 生成 float ****weights_array\n')
            fc.write('\t'+D_or_F+' ****{0}_pw;\n'.format(layer.name))
            fc.write('\t{0}_pw = alloc_4D({1}, 1, 1, {2});\n'.format(layer.name, layer.units, layer.input_shape[1]))
            fc.write('\tfor(int i = 0; i < {0}; i++){{\n'.format(layer.units))
            fc.write('\t\tfor(int j = 0; j < {0}; j++){{\n'.format(layer.input_shape[1]))
            fc.write('\t\t\t\t{0}_pw[i][0][0][j] = {0}_weights[i][j];\n'.format(layer.name))
            fc.write('\t\t}\n')
            fc.write('\t}\n')
            # 创建一个全连接层
            fc.write('\tDenseLayer *_{0};\n\t_{0} = new_Dense({1}, 1, 1, {2}, {0}_pw, &{0}_biases);\n'.format(layer.name, layer.units, layer.input_shape[1]))
        # BatchNormalization层,只有权重~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif isinstance(layer, keras.layers.BatchNormalization):
            weights = layer.get_weights()            

            if len(weights)==4:  # 未忽略gama
                # gama*************************************
                fh.write(D_or_F+" {0}_gamma[1][{1}]={{".format(layer.name, len(weights[0])))
                for n in range(len(weights[0])):
                    fh.write(format(weights[0][n], jing_du) + d_or_f+",")
                fh.write('};\n')
                # beta*************************************
                fh.write(D_or_F+" {0}_beta[1][{1}]={{".format(layer.name, len(weights[1])))
                for n in range(len(weights[1])):
                    fh.write(format(weights[1][n], jing_du) + d_or_f+",")
                fh.write('};\n')
                # moving_mean*************************************
                fh.write(D_or_F+" {0}_moving_mean[1][{1}]={{".format(layer.name, len(weights[2])))
                for n in range(len(weights[2])):
                    fh.write(format(weights[2][n], jing_du) + d_or_f+",")
                fh.write('};\n')
                # moving_variance*************************************
                fh.write(D_or_F+" {0}_moving_variance[1][{1}]={{".format(layer.name, len(weights[3])))
                for n in range(len(weights[3])):
                    fh.write(format(weights[3][n], jing_du) + d_or_f+",")
                fh.write('};\n')
            elif len(weights)==3:  #  忽略gama，即gama默认为1
                # gama*************************************
                fh.write(D_or_F+" {0}_gamma[1][{1}]={{".format(layer.name, len(weights[0])))
                for n in range(len(weights[0])):
                    fh.write(format(1, jing_du) + d_or_f+",")
                fh.write('};\n')
                # beta*************************************
                fh.write(D_or_F+" {0}_beta[1][{1}]={{".format(layer.name, len(weights[0])))
                for n in range(len(weights[0])):
                    fh.write(format(weights[0][n], jing_du) + d_or_f+",")
                fh.write('};\n')
                # moving_mean*************************************
                fh.write(D_or_F+" {0}_moving_mean[1][{1}]={{".format(layer.name, len(weights[1])))
                for n in range(len(weights[1])):
                    fh.write(format(weights[1][n], jing_du) + d_or_f+",")
                fh.write('};\n')
                # moving_variance*************************************
                fh.write(D_or_F+" {0}_moving_variance[1][{1}]={{".format(layer.name, len(weights[2])))
                for n in range(len(weights[2])):
                    fh.write(format(weights[2][n], jing_du) + d_or_f+",")
                fh.write('};\n')
            

    fc.write('\n\n\t/******************  前向传播  ******************/\n')
    for layer in model.layers:
        if isinstance(layer, keras.engine.input_layer.InputLayer):  # 跳过输入层
            continue
        fc.write('\tTensor* {0};\n'.format(layer.name))
        # 卷积层计算
        if isinstance(layer, keras.layers.convolutional.Conv1D) or isinstance(layer, keras.layers.convolutional.Conv2D):
            last_layer = (layer.input.name).split('/')[0]
            if layer.get_config()['activation'].upper() == "RELU":
                activation = "ReLU_activation"
            elif layer.get_config()['activation'].upper() == "LINEAR":
                activation = "linear_activation"
            elif layer.get_config()['activation'].upper() == "SIGMOID":
                activation = "sigmoid_activation"
            elif layer.get_config()['activation'].upper() == "SOFTMAX":
                activation = "softmax_activation"
            elif layer.get_config()['activation'].upper() == "ELU":
                activation = "ELU_activation"
            else:
                print('【未实现的激活函数】：',layer.get_config()['activation'])
            fc.write('\t{0} = Conv({2}, _{0}, {1}, 0);\n'.format(layer.name, activation, last_layer))
        elif isinstance(layer, keras.layers.core.Dense):
            last_layer = (layer.input.name).split('/')[0]
            if layer.get_config()['activation'].upper() == "RELU":
                activation = "ReLU_activation"
            elif layer.get_config()['activation'].upper() == "LINEAR":
                activation = "linear_activation"
            elif layer.get_config()['activation'].upper() == "SIGMOID":
                activation = "sigmoid_activation"
            elif layer.get_config()['activation'].upper() == "SOFTMAX":
                activation = "softmax_activation"
            elif layer.get_config()['activation'].upper() == "ELU":
                activation = "ELU_activation"
            else:
                print('【未实现的激活函数】：',layer.get_config()['activation'])
            fc.write('\t{0} = Dense({2}, _{0}, {1}, 0);\n'.format(layer.name, activation, last_layer))
        elif isinstance(layer, keras.layers.pooling.MaxPooling2D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{0} = MaxPool({1}, {4}, {2}, {3}, {5}, {6}, 0);\n'.format(layer.name, last_layer, layer.pool_size[0], layer.strides[0], layer.pool_size[1], layer.strides[1], layer.padding.upper()))
        elif isinstance(layer, keras.layers.pooling.MaxPooling1D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{4} = MaxPool({3}, 1, {0}, {1}, 1, {2}, 0);\n'.format(layer.pool_size[0], layer.strides[0], layer.padding.upper(), last_layer, layer.name))
        elif isinstance(layer, keras.layers.core.Flatten):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{0} = FlattenW({1}, 0);\n'.format(layer.name, last_layer))
        elif isinstance(layer, keras.layers.convolutional.UpSampling2D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{0} = UpSample({1}, {2}, {3}, 0);\n'.format(layer.name, last_layer, layer.size[0], layer.size[1]))
        elif isinstance(layer, keras.layers.convolutional.UpSampling1D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{1} = UpSample({2}, 1, {0}, 0);\n'.format(layer.size, layer.name, last_layer))
        elif isinstance(layer, keras.layers.normalization.batch_normalization.BatchNormalization):
            last_layer = (layer.input.name).split('/')[0]
            weights = layer.get_weights()
            if len(weights)==4:  # 未忽略gama
                # gama*************************************
                fc.write("\t"+D_or_F+"(*{0}_pg)[{1}];\n".format(layer.name, len(weights[0])))
                fc.write("\t{0}_pg={0}_gamma;\n".format(layer.name))
                # beta*************************************
                fc.write("\t"+D_or_F+"(*{0}_pb)[{1}];\n".format(layer.name, len(weights[1])))
                fc.write("\t{0}_pb={0}_beta;\n".format(layer.name))
                # moving_mean******************************
                fc.write("\t"+D_or_F+"(*{0}_pm)[{1}];\n".format(layer.name, len(weights[2])))
                fc.write("\t{0}_pm={0}_moving_mean;\n".format(layer.name))
                # moving_variance**************************
                fc.write("\t"+D_or_F+"(*{0}_pv)[{1}];\n".format(layer.name, len(weights[3])))
                fc.write("\t{0}_pv={0}_moving_variance;\n".format(layer.name)) 
                fc.write('\t{0} = BatchNormalization({1}, &{0}_pg, &{0}_pb, &{0}_pm, &{0}_pv, 0);\n'.format(layer.name, last_layer))
            elif len(weights)==3:  #  忽略gama，即gama默认为1
                # gama*************************************
                fc.write("\t"+D_or_F+"(*{0}_pg)[{1}];\n".format(layer.name, len(weights[0])))
                fc.write("\t{0}_pg={0}_gamma;\n".format(layer.name))
                # beta*************************************
                fc.write("\t"+D_or_F+"(*{0}_pb)[{1}];\n".format(layer.name, len(weights[0])))
                fc.write("\t{0}_pb={0}_beta;\n".format(layer.name))
                # moving_mean******************************
                fc.write("\t"+D_or_F+"(*{0}_pm)[{1}];\n".format(layer.name, len(weights[1])))
                fc.write("\t{0}_pm={0}_moving_mean;\n".format(layer.name))
                # moving_variance**************************
                fc.write("\t"+D_or_F+"(*{0}_pv)[{1}];\n".format(layer.name, len(weights[2])))
                fc.write("\t{0}_pv={0}_moving_variance;\n".format(layer.name)) 
                fc.write('\t{0} = BatchNormalization({1}, &{0}_pg, &{0}_pb, &{0}_pm, &{0}_pv, 0);\n'.format(layer.name, last_layer))
        elif isinstance(layer, keras.layers.merge.Add):
            length = len(layer.input)
            layer1 = (layer.input[0].name).split('/')[0]
            layer2 = (layer.input[1].name).split('/')[0]
            fc.write('\t{0} = Add({1}, {2}, 0);\n'.format(layer.name, layer1, layer2))
            if length > 2:
                for i in range(2,length):
                    fc.write('\t{0} = Add({0},{1},0);\n'.format(layer.name, (layer.input[i].name).split('/')[0]))   
        elif isinstance(layer, keras.layers.core.Dropout):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{0} = {1};\n'.format(layer.name, last_layer))
        elif isinstance(layer, keras.layers.merge.Concatenate):
            length = len(layer.input)
            layer1 = (layer.input[0].name).split('/')[0]
            layer2 = (layer.input[1].name).split('/')[0]
            fc.write('\t{0} = Concatenate({1},{2},0);\n'.format(layer.name, layer1, layer2))
            if length > 2:
                for i in range(2,length):
                    fc.write('\t{0} = Concatenate({0},{1},0);\n'.format(layer.name, (layer.input[i].name).split('/')[0]))
        elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{0} = GlobalAveragePooling({1}, 2, 0);\n'.format(layer.name, last_layer))
        elif isinstance(layer, keras.layers.GlobalAveragePooling1D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{0} = GlobalAveragePooling({1}, 1, 0);\n'.format(layer.name, last_layer))
        elif isinstance(layer, keras.layers.pooling.AveragePooling2D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{0} = AveragePool({1}, {4}, {2}, {3}, {5}, {6}, 0);\n'.format(layer.name, last_layer, layer.pool_size[0], layer.strides[0], layer.pool_size[1], layer.strides[1], layer.padding.upper()))
        elif isinstance(layer, keras.layers.pooling.AveragePooling1D):
            last_layer = (layer.input.name).split('/')[0]
            fc.write('\t{4} = AveragePool({3}, 1, {0}, {1}, 1, {2}, 0);\n'.format(layer.pool_size[0], layer.strides[0], layer.padding.upper(), last_layer, layer.name))
        elif isinstance(layer, keras.layers.core.Activation):
            last_layer = (layer.input.name).split('/')[0]
            if layer.get_config()['activation'].upper() == "RELU":
                fc.write('\t{0} = ReLU_activation({1}, 0);\n'.format(layer.name, last_layer))
            elif layer.get_config()['activation'].upper() == "LINEAR":
                fc.write('\t{0} = linear_activation({1}, 0);\n'.format(layer.name, last_layer))
            elif layer.get_config()['activation'].upper() == "SIGMOID":
                fc.write('\t{0} = sigmoid_activation({1}, 0);\n'.format(layer.name, last_layer))
            elif layer.get_config()['activation'].upper() == "SOFTMAX":
                fc.write('\t{0} = softmax_activation({1}, 0);\n'.format(layer.name, last_layer)) 
            elif layer.get_config()['activation'].upper() == "ELU":
                fc.write('\t{0} = ELU_activation({1}, 0);\n'.format(layer.name, last_layer))
            else:
                print('【未实现的激活函数】：',layer.get_config()['activation'])
        else:
            print('【未实现网络层结构】：', layer.name,type(layer))
    fc.write('\n\n\t//print_tensor({0});\n'.format((model.layers[0].input.name).split('/')[0]))
    fc.write('\n\n\t/******************  释放内存  ******************/\n')
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv1D) or isinstance(layer, keras.layers.convolutional.Conv2D):
            fc.write("\tfree_ConvLayer(_{0});\n".format(layer.name))
        elif isinstance(layer, keras.layers.core.Dense):
            fc.write("\tfree_DenseLayer(_{0});\n".format(layer.name))
    fh.write('#endif')
    fh.close()
    fc.write('\n\tprintf("\\nHello World!\\n");\n')
    fc.write('\treturn 0;\n}')
    fc.close()