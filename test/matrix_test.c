#include <matrix.h>

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

int main(){
	float A[] = {
		1.3, 3.21, 9.48, 0.13,
		2.5, 2.44, 0.58, 6.34,
		5.9, 1.78, 8.37, 4.44
	};
	float b[] = {4.23, 5.14, 7.13, 0.55};
	print_matrix(A,3,4);
	print_matrix(b,4,1);
	float r[3] = {0};

	fmatrix_mul_v(r,A,b,3,4);
	print_matrix(r,3,1);

	fvector_mul_vt(A,b,r,4,3);
	print_matrix(A,4,3);

	fmatrix_t_mul_v(r,A,b,4,3);
	print_matrix(r,3,1);
	return 0;
}