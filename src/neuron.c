#include "neuron.h"

#include <stdlib.h>

FVECTOR forward(layer_t* l, FVECTOR input) {
    l->input = input;
    fmatrix_mul_v(l->output, l->W, l->input, l->row, l->col);
    fmatrix_add(l->output, l->output, l->b, l->output_num, 1);
    l->activation_func->activation_func(l->output, l->output_num);
    return l->output;
}

void error_backward(layer_t* l, FVECTOR err, FVECTOR input_err) {
    fmatrix_t_mul_v(input_err, l->W, err, l->row, l->col);
    l->previous->activation_func->activate_derivative(input_err, l->previous->output, l->input_num);
}

void update_param(layer_t* l, FVECTOR err, float rate) {
    FMATRIX Werr = (FMATRIX)malloc(sizeof(float) * l->row * l->col);
    if (Werr == NULL) {
        exit(1);
    }

    /* 计算权重梯度并更新权重 */
    fvector_mul_vt(Werr, err, l->input, l->row, l->col);
    fmatrix_num(Werr, Werr, rate, l->row, l->col);
    fmatrix_sub(l->W, l->W, Werr, l->row, l->col);
    /* 重复利用Werr的内存空间，更新偏置 */
    fmatrix_num(Werr, err, rate, l->row, 1);
    fmatrix_sub(l->b, l->b, Werr, l->row, 1);

    free(Werr);
}
