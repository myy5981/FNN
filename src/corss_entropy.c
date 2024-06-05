#include "loss.h"

#include <math.h>

float corss_entropy_loss_func(FVECTOR output, FVECTOR real_value, int num) {
    for (int i = 0; i < num; i++) {
        if(real_value[i]!=0.0f){
            return -log(output[i]);
        }
    }
    return 0.0f;
}

void corss_entropy_loss_derivative(FVECTOR loss_err, FVECTOR output, FVECTOR real_value, int num) {
    for (int i = 0; i < num; i++) {
        if(real_value[i]!=0.0f) {
            loss_err[i] = - 1.0f / output[i];
        }else{
            loss_err[i] = 0.0f;
        }
    }
}

static const loss_func_t corss_entropy_s = {
    .loss_func = corss_entropy_loss_func,
    .loss_derivative = corss_entropy_loss_derivative};

const loss_func_t* corss_entropy = &corss_entropy_s;
