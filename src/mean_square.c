#include "loss.h"

float mean_square_loss_func(FVECTOR output, FVECTOR real_value, int num) {
    float x = 0;
    for (int i = 0; i < num; i++) {
        x += (output[i] - real_value[i]) * (output[i] - real_value[i]);
    }
    return x / ((float)num);
}

void mean_square_loss_derivative(FVECTOR loss_err, FVECTOR output, FVECTOR real_value, int num) {
    for (int i = 0; i < num; i++) {
        loss_err[i] = 2.0f * (output[i] - real_value[i]) / ((float)num);
    }
}

static const loss_func_t mean_square_s = {
    .loss_func = mean_square_loss_func,
    .loss_derivative = mean_square_loss_derivative
};

const loss_func_t* mean_square = &mean_square_s;
