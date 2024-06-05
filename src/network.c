#include "network.h"

#include <stdlib.h>

fnn_t* new_fnn(float rate,const loss_func_t* loss) {
    fnn_t* fnn = (fnn_t*)malloc(sizeof(fnn_t));
    if (fnn == NULL) {
        exit(1);
    }
    fnn->head = NULL;
    fnn->tail = NULL;
    fnn->layers = 0;
    fnn->rate = rate;
    fnn->loss = loss;
    return fnn;
}

void fnn_add_layer(fnn_t* fnn, int input_num, int output_num,const activation_func_t* act_func) {
    /* 分配初始内存 */
    layer_t* l = (layer_t*)malloc(sizeof(layer_t));
    if (l == NULL) {
        exit(1);
    }
    l->input = NULL;
    l->input_num = input_num;
    l->output_num = output_num;
    l->activation_func = act_func;
    l->W = (FMATRIX)malloc(sizeof(float) * input_num * output_num);
    if (l->W == NULL) {
        exit(1);
    }
    l->b = (FVECTOR)malloc(sizeof(float) * output_num);
    if (l->b == NULL) {
        exit(1);
    }
    l->output = (FVECTOR)malloc(sizeof(float) * output_num);
    if (l->output == NULL) {
        exit(1);
    }
    l->activation_func->weight_init(l->W, l->b, l->row, l->col);
    /* 双向链表尾插入 */
    if (fnn->tail == NULL) {
        fnn->head = l;
        fnn->tail = l;
        l->next = NULL;
        l->previous = NULL;
    } else {
        fnn->tail->next = l;
        l->previous = fnn->tail;
        l->next = NULL;
        fnn->tail = l;
    }
    fnn->layers++;
}

FVECTOR fnn_forward(fnn_t* fnn, FVECTOR input) {
    layer_t* l = fnn->head;
    while (l != NULL) {
        input = forward(l, input);
        l = l->next;
    }
    return input;
}

void fnn_backward(fnn_t* fnn, FVECTOR real_output) {
    layer_t* l = fnn->tail;
    if (l == NULL) {
        return;
    }
    FVECTOR err = (FVECTOR)malloc(sizeof(float) * l->output_num);
    if (err == NULL) {
        exit(1);
    }
    fnn->loss->loss_derivative(err, l->output, real_output, l->output_num);
    l->activation_func->activate_derivative(err, l->output, l->output_num);
    do {
        /* 对于第一隐藏层，不继续误差传播 */
        if (l->previous == NULL) {
            update_param(l, err, fnn->rate);
            break;
        }
        FVECTOR pre_err = (FVECTOR)malloc(sizeof(float) * l->input_num);
        if (pre_err == NULL) {
            exit(1);
        }
        error_backward(l, err, pre_err);
        update_param(l, err, fnn->rate);
        free(err);
        err = pre_err;
        l = l->previous;
    } while (l != NULL);
    free(err);
}

void fnn_serialize(fnn_t* fnn, const char* path) {}

fnn_t* fnn_deserialize(const char* path) {
    return NULL;
}
