#include "matrix.h"

#include <stdio.h>

void print_matrix(FMATRIX m, int row, int col){
	for (int i = 0; i < row*col; i++) {
		printf("%2.2f ",m[i]);
		if(i%col == col-1){
			printf("\n");
		}
	}
	printf("\n");
}

void fmatrix_mul_v(FVECTOR r, FMATRIX A, FVECTOR b, int row, int col) {
    for (int i = 0; i < row; i++) {
        float ri = 0;
        for (int j = 0; j < col; j++) {
            ri += (A[i * col + j] * b[j]);
        }
        r[i] = ri;
    }
}

void fmatrix_t_mul_v(FVECTOR r, FMATRIX A, FVECTOR b, int row, int col) {
    for (int i = 0; i < col; i++) {
        float ri = 0;
        for (int j = 0; j < row; j++) {
            ri += A[j * col + i] * b[j];
        }
        r[i] = ri;
    }
}

void fvector_mul_vt(FMATRIX R, FVECTOR a, FVECTOR b, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            R[i * col + j] = a[i] * b[j];
        }
    }
}
