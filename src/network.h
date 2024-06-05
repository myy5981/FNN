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

fnn_t* new_fnn(float rate,const loss_func_t* loss);

void fnn_add_layer(fnn_t* fnn, int input_num, int output_num,const activation_func_t* act_func);

FVECTOR fnn_forward(fnn_t* fnn, FVECTOR input);

void fnn_backward(fnn_t* fnn, FVECTOR real_output);

void fnn_serialize(fnn_t* fnn, const char* path);

fnn_t* fnn_deserialize(const char* path);

#endif
