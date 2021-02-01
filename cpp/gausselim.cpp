#include "gausselim.hpp"


#define A(I, J) (A_[I * N + J])
#define B(I, J) (B_[I * M + J])
void gauss_elim(int N, int M, float* restrict A_, float* restrict B_) {
	for (int k = 0; k < N - 1; k++) {
		for (int i = k + 1; i < N; i++) {
			A(i, k) /= A(k, k);
		}

		for (int i = k + 1; i < N; i++) {
			for (int j = k + 1; j < N; j++) {
				A(i, j) -= A(i, k) * A(k, j);
			}
		}

		for (int i = k + 1; i < N; i++) {
			for (int j = 0; j < M; j++) {
				B(i, j) -= A(i, k) * B(k, j);
			}
		}
	}

	for (int k = N - 1; k >= 0; k--) {
		for (int j = N - 1; j >= k + 1; j--) {
			for (int i = 0; i < M; i++) {
				B(k, i) -= A(k, j) * B(j, i);
			}
		}

		for (int i = 0; i < M; i++) {
			B(k, i) /= A(k, k);
		}
	}
}