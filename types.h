#ifndef TYPES_H
#define TYPES_H

#ifndef __OPENCL__
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <errno.h>

#define __global
#define __constant

#else /* __OPENCL__ */

#define sinf    sin
#define cosf    cos
#define tanf    tan
#define acosf   acos
#define atan2f  atan2
#define fabsf   fabs
#define sqrtf   sqrt
#define powf    pow
#define floorf  floor

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

static inline float clamp(float lo, float hi, float v)
{
	return MAX(lo, MIN(hi, v));
}

static inline uint32_t fls_bit(uint64_t x)
{
	return x ? sizeof(x) * 8 - __builtin_clzl(x) : 0;
}

#define ffs_bit(x)	\
	(sizeof(x) == 8 ? ffsl(x) : ffs(x))

static inline uint64_t atomic64_cmpxchg(uint64_t *p, uint64_t old, uint64_t new)
{
	return __sync_val_compare_and_swap(p, old, new);
}

static inline uint32_t atomic64_dec(uint64_t *p)
{
	return __sync_fetch_and_sub(p, 1);
}

static inline uint32_t atomic64_inc(uint64_t *p)
{
	return __sync_fetch_and_add(p, 1);
}

static inline uint32_t atomic32_dec(uint32_t *p)
{
	return __sync_fetch_and_sub(p, 1);
}

static inline uint32_t atomic32_inc(uint32_t *p)
{
	return __sync_fetch_and_add(p, 1);
}

#define memcpy_from_global memcpy
#define memcpy_to_global memcpy

static inline int get_alloc_hint(void)
{
	/* Can be any rand, 0 for now */
	return 0;
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
	return atom_cmpxchg((__global long *)p, old, new);
}

static inline uint32_t atomic64_dec(__global uint64_t *p)
{
	return atom_dec((__global long *)p);
}

static inline uint32_t atomic64_inc(__global uint64_t *p)
{
	return atom_inc((__global long *)p);
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

static inline bool is_power_of_two(uint64_t n)
{
	return (n != 0 && ((n & (n - 1)) == 0));
}

static inline uint32_t ilog2(uint64_t n)
{
	return fls_bit(n) - 1;
}

static inline uint64_t roundup_power_of_two(uint64_t n)
{
	return 1UL << fls_bit(n - 1);
}

static inline float modulo(float f)
{
    return f - floorf(f);
}

static inline float deg2rad(float deg)
{
	return deg * M_PI / 180;
}

#endif /* TYPES_H */
