#define export 
#define uniform
#define restrict // TODO: is this allowed in C++??

export void gauss_elim(int N, int M, float* restrict A, float* restrict B);