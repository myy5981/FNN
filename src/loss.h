#ifndef MYY_FNN_LOSS_H
#define MYY_FNN_LOSS_H

#include "matrix.h"

typedef struct loss_func_s{
    /**
     * 指定num维的输出数据与对应的真实数据，计算误差值
    */
    float (*loss_func)(FVECTOR output, FVECTOR  real_value, int num);
    /**
     * 指定num维的输出数据与对应的真实数据，计算误差梯度
    */
    void (*loss_derivative)(FVECTOR loss_err, FVECTOR output, FVECTOR real_value, int num);
} loss_func_t;

/* 误差函数通过导出结构体指针在此注册 */

#define MEAN_SQUARE mean_square
extern const loss_func_t* mean_square;

#define CORSS_ENTROPY corss_entropy
extern const loss_func_t* corss_entropy;

#endif
