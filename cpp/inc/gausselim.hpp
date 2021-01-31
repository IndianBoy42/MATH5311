#define export
#define uniform
#define restrict  // TODO: is this allowed in C++??

export void gauss_elim(uniform int N, uniform int M, uniform float* restrict A, uniform float* restrict B);