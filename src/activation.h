#ifndef MYY_FNN_ACTIVATION_H
#define MYY_FNN_ACTIVATION_H

#include "matrix.h"

typedef struct activation_func_s {
    /* 激活函数，结果直接更新在输入中 */
    void (*activation_func)(FVECTOR linear_output, int row);
    /* 激活函数的导数，该函数传入输出误差与输出值，计算线性输出的误差，更新在传入的输出误差中 */
    void (*activate_derivative)(FVECTOR err, FVECTOR output, int row);
    /* 该激活函数对应的权重矩阵初始化方案 */
    void (*weight_init)(FMATRIX W, FVECTOR b, int row, int col);
} activation_func_t;

/* 激活函数通过导出结构体指针在此注册 */

#define SOFTMAX softmax
extern const activation_func_t* softmax;

#define RELU relu
extern const activation_func_t* relu;

#endif
