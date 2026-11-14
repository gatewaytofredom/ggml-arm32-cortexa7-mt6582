#pragma once
#include <sys/mman.h>
#ifndef POSIX_MADV_NORMAL
#define POSIX_MADV_NORMAL MADV_NORMAL
#define POSIX_MADV_RANDOM MADV_RANDOM
#define POSIX_MADV_SEQUENTIAL MADV_SEQUENTIAL
#define POSIX_MADV_WILLNEED MADV_WILLNEED
#define POSIX_MADV_DONTNEED MADV_DONTNEED
#endif
static inline int posix_madvise(void * addr, size_t len, int advice) {
    return madvise(addr, len, advice);
}
