#ifndef MYY_FNN_MATRIX_H
#define MYY_FNN_MATRIX_H

typedef float* FMATRIX;
typedef float* FVECTOR;

void print_matrix(FMATRIX m, int row, int col);
/**
 * 计算矩阵A与向量b的乘积保存在r中：r = A * b
 * 后两个参数指定矩阵的维度，A为为row行col列矩阵，b为col维列向量
 * 计算结果r为row维列向量，调用者应确保R有足够的空间，以免产生溢出
 * 调用fmatrix_mul_v(r,A,b,row,col)等价于调用fmatrix_mul(r,A,b,row,col,1)
 */
void fmatrix_mul_v(FVECTOR r, FMATRIX A, FVECTOR b, int row, int col);

/**
 * 计算矩阵A的转置与向量b的乘积保存在r中：r = A^T * b
 * 后两个参数指定矩阵的维度，A为为row行col列矩阵，b为row维列向量
 * 计算结果r为col维列向量，调用者应确保R有足够的空间，以免产生溢出
 */
void fmatrix_t_mul_v(FVECTOR r, FMATRIX A, FVECTOR b, int row, int col);

/**
 * 计算向量a乘以向量b的转置保存在矩阵R中：R = a * b^T
 * 后两个参数指定向量的维度，a为row维列向量，b为col维列向量
 * 计算结果R为row行col列矩阵，调用者应确保R有足够的空间，以免产生溢出
 */
void fvector_mul_vt(FMATRIX R, FVECTOR a, FVECTOR b, int row, int col);

/**
 * 计算A中每个元素乘以n保存在R中：R = nA
 * 后两个参数指定矩阵的维度，R、A均为为row行col列矩阵
 * 调用者应确保R有足够的空间，以免产生溢出
 */
#define fmatrix_num(R, A, n, row, col)            \
    do {                                          \
        for (int i = 0; i < (row) * (col); ++i) { \
            (R)[i] = (n) * (A)[i];                \
        }                                         \
    } while (0)

/**
 * 计算A与B的和保存在R中：R = A + B
 * 后两个参数指定矩阵的维度，R、A、B均为为row行col列矩阵
 * 调用者应确保R有足够的空间，以免产生溢出
 */
#define fmatrix_add(R, A, B, row, col)            \
    do {                                          \
        for (int i = 0; i < (row) * (col); ++i) { \
            (R)[i] = (A)[i] + (B)[i];             \
        }                                         \
    } while (0)

/**
 * 计算A与B的差保存在R中：R = A - B
 * 后两个参数指定矩阵的维度，R、A、B均为为row行col列矩阵
 * 调用者应确保R有足够的空间，以免产生溢出
 */
#define fmatrix_sub(R, A, B, row, col)            \
    do {                                          \
        for (int i = 0; i < (row) * (col); ++i) { \
            (R)[i] = (A)[i] - (B)[i];             \
        }                                         \
    } while (0)

#endif
