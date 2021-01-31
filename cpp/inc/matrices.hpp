#include <iomanip>
#include <iostream>

void print_matrix(auto* A, int N, int M) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			std::cout << std::setw(10) << A[i * N + j] << '\t';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}