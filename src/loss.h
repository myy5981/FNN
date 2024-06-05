#ifndef MYY_FNN_LOSS_H
#define MYY_FNN_LOSS_H

#include "matrix.h"

typedef struct loss_func_s{
    float (*loss_func)(FVECTOR, FVECTOR, int);
    void (*loss_derivative)(FVECTOR, FVECTOR, FVECTOR, int);
} loss_func_t;

#define MEAN_SQUARE mean_square
extern const loss_func_t* mean_square;

#define CORSS_ENTROPY corss_entropy
extern const loss_func_t* corss_entropy;

#endif
