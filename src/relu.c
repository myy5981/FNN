#include "activation.h"
#include "random.h"

#include <math.h>

void relu_activation_func(FVECTOR linear_output, int row) {
    for (int i = 0; i < row; i++) {
        if (linear_output[i] < 0.0f) {
            linear_output[i] = 0.0f;
        }
    }
}

void relu_activation_derivative(FVECTOR err, FVECTOR output, int row) {
    for (int i = 0; i < row; i++) {
        if (output[i] <= 0.0f) {
            err[i] = 0.0f;
        }
    }
}

void relu_weight_init(FMATRIX W, FVECTOR b, int row, int col){
    float stddev = sqrtf(2.0f / ((float)col));
    for (int i = 0; i < row*col; i++) {
        W[i] = randomf(0,stddev);
    }
    for (int i = 0; i < row; i++) {
        b[i] = 0;
    }
}

static const activation_func_t relu_s = {
    .activation_func = relu_activation_func,
    .activate_derivative = relu_activation_derivative,
    .weight_init = relu_weight_init
};

const activation_func_t* relu = &relu_s;
