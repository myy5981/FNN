#include "activation.h"
#include "random.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define E 2.7182818284f

void softmax_activation_func(FVECTOR linear_output, int row) {
    float fenmu = 0;
    for (int i = 0; i < row; i++) {
        fenmu += powf(E, linear_output[i]);
    }
    for (int i = 0; i < row; i++) {
        linear_output[i] = powf(E, linear_output[i]) / fenmu;
    }
}

void softmax_activation_derivative(FVECTOR err, FVECTOR output, int row) {
    FVECTOR _tmp = (FVECTOR)malloc(sizeof(float) * row);
    if (_tmp == NULL) {
        exit(1);
    }
    for (int i = 0; i < row; i++) {
        float erri = 0;
        for (int j = 0; j < row; j++) {
            erri -= output[i] * output[j] * err[j];
            if (i == j) {
                erri += output[i] * err[j];
            }
        }
        _tmp[i] = erri;
    }
    memcpy(err, _tmp, sizeof(float) * row);
    free(_tmp);
}

void softmax_weight_init(FMATRIX W, FVECTOR b, int row, int col){
    float stddev = sqrtf(2.0f / ((float)col));
    for (int i = 0; i < row*col; i++) {
        W[i] = randomf(0,stddev);
    }
    for (int i = 0; i < row; i++) {
        b[i] = 0;
    }
}

static const activation_func_t softmax_s = {
    .activation_func = softmax_activation_func,
    .activate_derivative = softmax_activation_derivative,
    .weight_init = softmax_weight_init
};

const activation_func_t* softmax = &softmax_s;
