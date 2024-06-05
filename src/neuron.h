#ifndef MYY_FNN_NEURON_H
#define MYY_FNN_NEURON_H

#include "activation.h"
#include "matrix.h"

typedef struct layer_s {
    /* 层输入，该字段初始化时不分配内存 */
    FVECTOR input;
    /* 层输出 */
    FVECTOR output;
    /* 权重矩阵 */
    FMATRIX W;
    /* 偏置向量 */
    FVECTOR b;
    /* 输入数据规模，也是权重矩阵的列数 */
    union {
        int input_num;
        int col;
    };
    /* 输出数据，规模，也是权重矩阵的行数 */
    union {
        int output_num;
        int row;
    };
    /* 激活函数 */
    const activation_func_t* activation_func;

    struct layer_s* next;
    struct layer_s* previous;
} layer_t;

/**
 * 正向传播，input为该层的输入
 * 对于第一隐藏层，该参数为神经网络的输入
 * 对于后续的隐藏层和输出层，该参数为上一层的输出
 * 返回该层的输出
 */
FVECTOR forward(layer_t* l, FVECTOR input);

/**
 * 误差反向传播
 * 指定该层线性输出的误差值err，计算上一层线性输出的误差值input_err
 * **注意：误差传播过程中需要使用权重矩阵，故调用该函数需要在update_param函数之前**
 */
void error_backward(layer_t* l, FVECTOR err, FVECTOR input_err);

/**
 * 根据该层线性输出误差与学习率，计算权重与偏置的改正并更新之
 * **注意：如需要进行误差反向传播，则调用error_backward需在调用本函数之前**
 */
void update_param(layer_t* l, FVECTOR err, float rate);

#endif
