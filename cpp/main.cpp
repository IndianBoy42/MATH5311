#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gausselim.hpp"
#include "gausselim_ispc.h"
#include "matrices.hpp"
#include "rel_assert.h"

// constexpr int N = 3;
// std::array<float, N *N> exampleA = {
// 	2,	1, -1,	//
// 	4,	5, -3,	//
// 	-2, 5, -2,	//
// };
// constexpr int N = 4;
// const std::array<float, N *N> srcMat = {
// 	3,	5, 7, 2,   //.
// 	1,	4, 7, 2,   //.
// 	6,	3, 9, 17,  //.
// 	13, 5, 4, 16,  //.
// };
constexpr int N = 1000;
const std::array<float, N *N> srcMat = {
#include "output.hmat"
};
std::array<float, N *N> exampleA = srcMat;
std::array<float, N *N> exampleB = srcMat;

constexpr int BENCH_ITERS = 10;

int main(int argc, char const *argv[]) {
	std::cout << "Lets-a-go" << std::endl;

	// gauss_elim(N, N, exampleA.data(), exampleB.data());
	ispc::gauss_elim(N, N, exampleA.data(), exampleB.data());
// print_matrix(exampleA.data(), N, N);
// print_matrix(exampleB.data(), N, N);
#if 0
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++) {
			if (i != j) {
				const auto zerook = std::abs(exampleB[i * N + j] - 0) < 1e-2;
				if (!zerook) {
					std::cout << exampleB[i * N + j] << std::endl;
				}
				rel_assert(zerook);
			} else {
				const auto oneok = std::abs(exampleB[i * N + j] - 1) < 1e-3;
				if (!oneok) {
					std::cout << exampleB[i * N + j] << std::endl;
				}
				rel_assert(oneok);
			}
		}
	}
#endif

	double avg = 0;
	for (int i = 0; i < BENCH_ITERS; i++) {
		exampleA = srcMat;
		exampleB = srcMat;
		auto start = std::chrono::high_resolution_clock::now();

		gauss_elim(N, N, exampleA.data(), exampleB.data());
		// ispc::gauss_elim(N, N, exampleA.data(), exampleB.data());

		std::chrono::duration<double> dur = std::chrono::high_resolution_clock::now() - start;
		// std::cout << dur.count() << std::endl;
		avg += ((double)dur.count()) / BENCH_ITERS;
	}
	std::cout  << "avg: "<< avg << std::endl;

	return 0;
}
