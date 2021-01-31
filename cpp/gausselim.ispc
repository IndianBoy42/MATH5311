#define restrict 

#define A(I, J) (A[I * N + J])
#define B(I, J) (B[I * M + J])
export void gauss_elim(uniform int N, uniform int M, uniform float A[], uniform float B[]) {
	for (int k = 0; k < N - 1; k++) {
		foreach(i = 0...N) {
			if (i <= k) continue;
			A(i, k) /= A(k, k);
		}

		for (int i = k+1; i < N; i++) {
			// for (int j = k+1; j < N; j++) {
			// foreach(j = (k+1)...N) {
			foreach(j = 0...N) {
				if (j <= k) continue;
				A(i, j) -= A(i, k) * A(k, j);
			}
		}

		for (int i = k+1; i < N; i++) {
			foreach(j = 0...M) {
				B(i, j) -= A(i, k) * B(k, j);
			}
		}
	}

	
	// foreach_tiled(k_ = 0 ... N-1, i = 0 ... M ) {
	// 	int k = N-k_-1;
	// 	for (int j = N - 1; j >= k+1; j--) {
	// 		B(k, i) -= A(k, j) * B(j, i);
	// 	}
	// }
	// foreach_tiled(k_ = 0 ... N-1, i = 0 ... M ) {
	// 	int k = N-k_-1;
	// 	B(k, i) /= A(k, k);
	// }
	for (int k = N - 1; k >= 0; k--) {
		for (int j = N - 1; j >= k+1; j--) {
			foreach(i = 0...M){
				B(k, i) -= A(k, j) * B(j, i);
			}
		}
		foreach(i = 0...M){
			B(k, i) /= A(k, k);
		}
	}
}