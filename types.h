#ifndef TYPES_H
#define TYPES_H

#ifndef __OPENCL__
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <errno.h>
#include <assert.h>

#define __global
#define __constant

#ifdef __CUDACC__
/* Mark functions callable from both host and CUDA device */
#define __accelerated __device__ __host__
/*
 * fmaxf/fminf: define as ternary macros for both host and device passes.
 * The <math.h> declarations are not consistently marked __device__ across
 * CUDA versions, and the __fmaxf/__fminf intrinsics are __host__-only in
 * some releases, causing "calling __host__ function from __host__ __device__"
 * warnings.  The ternary is always valid in both compilation passes.
 */
#define fmaxf(a, b) ((a) >= (b) ? (a) : (b))
#define fminf(a, b) ((a) <= (b) ? (a) : (b))
/* Override __constant for device compilation: use CUDA constant memory */
#ifdef __CUDA_ARCH__
#undef __constant
#define __constant __constant__
#endif
#else
#define __accelerated
#endif

#else /* __OPENCL__ */
/* In OpenCL all static inline functions are device-callable by default */
#define __accelerated

/*
 * OpenCL C does not define 'typeof', but Clang (used by most OpenCL runtimes)
 * supports '__typeof__' as a GNU extension.  Map typeof to __typeof__ so that
 * shared macros (container_of, SWAP, ALIGN_PTR_*) compile without changes.
 */
#define typeof __typeof__

/*
 * Map *f math functions to their generic OpenCL equivalents, forcing the
 * float type explicitly.  A plain '#define sqrtf sqrt' leaves the compiler
 * to choose between float and double overloads from cl_kernel.h, causing
 * "call to 'sqrt' is ambiguous" errors.  The (float) cast removes the
 * ambiguity and matches what the *f suffix means on the host.
 */
#define sinf(x)      sin((float)(x))
#define cosf(x)      cos((float)(x))
#define tanf(x)      tan((float)(x))
#define acosf(x)     acos((float)(x))
#define atan2f(y, x) atan2((float)(y), (float)(x))
#define fabsf(x)     fabs((float)(x))
#define sqrtf(x)      sqrt((float)(x))
#define powf(x, y)    pow((float)(x), (float)(y))
#define floorf(x)     floor((float)(x))
#define fmaxf(x, y)   fmax((float)(x), (float)(y))
#define fminf(x, y)   fmin((float)(x), (float)(y))

typedef unsigned long  uint64_t;
typedef long           int64_t;
typedef unsigned int   uint32_t;
typedef int	           int32_t;
typedef unsigned short uint16_t;
typedef short          int16_t;
typedef unsigned char  uint8_t;
typedef char           int8_t;

#endif /* __OPENCL__ */

#ifndef offsetof
#define offsetof(t,m) __builtin_offsetof(t, m)
#endif

#ifndef container_of
#define container_of(ptr, type, member) ({			\
	const typeof( ((type*)0)->member )* __mptr = (ptr);	\
	(type*)( (uintptr_t)__mptr - offsetof(type, member));	\
})
#endif

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(*(x)))
#define ALIGN_UP(x, align_to)	(((x) + ((align_to)-1)) & ~((align_to)-1))
#define ALIGN_DOWN(x, align_to) ((x) & ~((align_to)-1))
#define ALIGN_PTR_UP(p, ptr_align_to)	\
	((typeof(p))ALIGN_UP((unsigned long)(p), ptr_align_to))
#define ALIGN_PTR_DOWN(p, ptr_align_to)	\
	((typeof(p))ALIGN_DOWN((unsigned long)(p), ptr_align_to))

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SWAP(a, b) do { typeof(a) temp = a; a = b; b = temp; } while (0)

#ifndef __OPENCL__

__accelerated static inline float clamp(float lo, float hi, float v)
{
	return MAX(lo, MIN(hi, v));
}

#ifdef __CUDACC__
__accelerated static inline uint32_t fls_bit(uint64_t x)
{
#ifdef __CUDA_ARCH__
	return x ? 64 - __clzll((unsigned long long)x) : 0;
#else
	return x ? sizeof(x) * 8 - __builtin_clzl(x) : 0;
#endif
}

__accelerated static inline int __ffs_bit(uint64_t x)
{
#ifdef __CUDA_ARCH__
	return __ffsll((unsigned long long)x);
#else
	return sizeof(x) == 8 ? ffsl(x) : ffs((int)x);
#endif
}
#define ffs_bit(x) __ffs_bit(x)
#else
static inline uint32_t fls_bit(uint64_t x)
{
	return x ? sizeof(x) * 8 - __builtin_clzl(x) : 0;
}
#define ffs_bit(x)	\
	(sizeof(x) == 8 ? ffsl(x) : ffs(x))
#endif /* __CUDACC__ */

__accelerated static inline uint64_t atomic64_cmpxchg(uint64_t *p, uint64_t old, uint64_t new)
{
#ifdef __CUDA_ARCH__
	return (uint64_t)atomicCAS((unsigned long long *)p,
				   (unsigned long long)old,
				   (unsigned long long)new);
#else
	return __sync_val_compare_and_swap(p, old, new);
#endif
}

__accelerated static inline uint32_t atomic64_dec(uint64_t *p)
{
#ifdef __CUDA_ARCH__
	return (uint32_t)atomicAdd((unsigned long long *)p,
				   (unsigned long long)-1LL);
#else
	return __sync_fetch_and_sub(p, 1);
#endif
}

__accelerated static inline uint32_t atomic64_inc(uint64_t *p)
{
#ifdef __CUDA_ARCH__
	return (uint32_t)atomicAdd((unsigned long long *)p, 1ULL);
#else
	return __sync_fetch_and_add(p, 1);
#endif
}

__accelerated static inline uint32_t atomic32_dec(uint32_t *p)
{
#ifdef __CUDA_ARCH__
	return atomicSub((unsigned int *)p, 1U);
#else
	return __sync_fetch_and_sub(p, 1);
#endif
}

__accelerated static inline uint32_t atomic32_inc(uint32_t *p)
{
#ifdef __CUDA_ARCH__
	return atomicAdd((unsigned int *)p, 1U);
#else
	return __sync_fetch_and_add(p, 1);
#endif
}

#define memcpy_from_global memcpy
#define memcpy_to_global memcpy

__accelerated static inline int get_alloc_hint(void)
{
#ifdef __CUDA_ARCH__
	return blockIdx.x * blockDim.x + threadIdx.x;
#else
	/* Can be any rand, 0 for now */
	return 0;
#endif
}

#else

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#define ENOENT  2
#define ENOMEM 12
#define EINVAL 22

#define ffs_bit(x) \
	((x) ? popcount((x) ^ ~-(x)) : 0)

static inline uint32_t fls_bit(uint64_t x)
{
	return x ? sizeof(x) * 8 - clz(x) : 0;
}

static inline uint64_t atomic64_cmpxchg(__global uint64_t *p, uint64_t old, uint64_t new)
{
	return atom_cmpxchg((__global ulong *)p, (ulong)old, (ulong)new);
}

static inline uint32_t atomic64_dec(__global uint64_t *p)
{
	return (uint32_t)atom_dec((__global ulong *)p);
}

static inline uint32_t atomic64_inc(__global uint64_t *p)
{
	return (uint32_t)atom_inc((__global ulong *)p);
}

static inline uint32_t atomic32_dec(__global uint32_t *p)
{
	return atomic_dec((__global int *)p);
}

static inline uint32_t atomic32_inc(__global uint32_t *p)
{
	return atomic_inc((__global int *)p);
}

static inline __global void *memset(__global void *p, int c, size_t n)
{
	__global char *xs = p;

	while (n--)
		*xs++ = c;
	return p;
}

static inline int memcmp(__global const void *cs, __global const void *ct,
			 size_t count)
{
	__global const unsigned char *su1, *su2;
	int res = 0;

	for (su1 = cs, su2 = ct; 0 < count; ++su1, ++su2, count--)
		if ((res = *su1 - *su2) != 0)
			break;
	return res;
}

static inline void memmove(__global void *dest, __global const char *src, size_t n)
{
	__builtin_memmove(dest, src, n);
}

static inline int get_alloc_hint(void)
{
	return get_global_id(0);
}

#define assert(exp) do {						\
	if (!(exp)) printf("Assert failed on %s:%d\n", __func__,  __LINE__); \
	} while (0)

#endif /* __OPENCL__ */

__accelerated static inline bool is_power_of_two(uint64_t n)
{
	return (n != 0 && ((n & (n - 1)) == 0));
}

__accelerated static inline uint32_t ilog2(uint64_t n)
{
	return fls_bit(n) - 1;
}

__accelerated static inline uint64_t roundup_power_of_two(uint64_t n)
{
	return 1UL << fls_bit(n - 1);
}

__accelerated static inline float modulo(float f)
{
    return f - floorf(f);
}

__accelerated static inline float deg2rad(float deg)
{
	return deg * M_PI / 180;
}

#endif /* TYPES_H */
