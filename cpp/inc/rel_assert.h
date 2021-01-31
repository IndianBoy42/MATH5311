#pragma once

#define rel_assert(EX) (void)((EX) || (__assert(#EX, __FILE__, __LINE__), 0))

#ifdef __cplusplus
extern "C" {
#endif

extern void __assert(const char *msg, const char *file, int line);

#ifdef __cplusplus
};
#endif