#ifndef MYY_FNN_NETWORK_H
#define MYY_FNN_NETWORK_H

#include "neuron.h"
#include "loss.h"

typedef struct fnn_s {
    float rate;
    int layers;
    layer_t* head;
    layer_t* tail;
    const loss_func_t* loss;
} fnn_t;

/**
 * 新建fnn，使用malloc分配内存
 * rate：学习率
 * loss：误差函数
*/
fnn_t* new_fnn(float rate,const loss_func_t* loss);

/**
 * 向fnn添加神经元层，其中使用malloc分配内存
 * input_num：输入规模
 * output_num：输出规模
 * act_func：激活函数
*/
void fnn_add_layer(fnn_t* fnn, int input_num, int output_num,const activation_func_t* act_func);

/**
 * 前向传播，返回指针指向输出层的输出，其内存为fnn_add_layer时分配的内存，无需free
 * input：输入数据
*/
FVECTOR fnn_forward(fnn_t* fnn, FVECTOR input);

/**
 * 误差反向传播，并更新权重和偏置
*/
void fnn_backward(fnn_t* fnn, FVECTOR real_output);

/**
 * 释放神经网络内存
*/
void fnn_destory(fnn_t* fnn);

#endif
