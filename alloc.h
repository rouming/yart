/**
 * Simple fixed-size thread-safe allocator. The whole memory buffer
 * (specified on alloc_init() call) is sliced on fixed-size chunks and
 * those chunks are managed by bitmap. Each alloc_chunk() call finds
 * first-set-bit in the free bitmap and atomically clears it, each
 * free_chunk() sets bit atomically, thus either alloc_chunk() or
 * free_chunk() can be called from different threads.
 *
 * 2021, Roman Penyaev <r.peniaev@gmail.com>
 */

#ifndef ALLOC_H
#define ALLOC_H

#ifndef __OPENCL__
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <errno.h>

static inline uint32_t fls_bit(uint64_t x)
{
	return x ? sizeof(x) * 8 - __builtin_clzl(x) : 0;
}

static inline uint32_t ffs_bit(uint64_t x)
{
	return ffsl(x);
}

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

typedef unsigned short uint16_t;

static inline uint32_t ffs_bit(uint64_t x)
{
	return x ? popcount(x ^ ~-x) : 0;
}

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

#endif

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)

enum {
	NO_FLAGS      = 0,
	FAIL_ON_NOMEM = 1<<0,
};

struct allocator {
	/* ulong requires support of cl_khr_int64_base_atomics */
	__global uint64_t *free_bitmap;
	__global void     *buffer;
	uint32_t nbytes_bitmap;
	uint32_t nlongs_bitmap;
	uint32_t chunk_shift;
	uint32_t chunk_size;
	uint32_t size;
	uint32_t chunks;
	uint32_t free_chunks;
	uint8_t  fail_on_nomem;
};

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

static inline void init_free_bitmap(__global struct allocator *a)
{
	uint32_t bitmap_size = a->size - (a->chunks << a->chunk_shift);
	uint32_t first_half = a->chunks >> 3;
	uint32_t last_half = bitmap_size - (round_up(a->chunks, 8) >> 3);
	__global uint8_t *bytes = (__global uint8_t *)a->free_bitmap;

	if (first_half)
		memset(a->free_bitmap, 0xff, first_half);
	if (a->chunks & 7)
		bytes[first_half] = (1<<(a->chunks % 8)) - 1;
	if (last_half)
		memset(bytes + first_half + 1, 0, last_half);
}

static inline int alloc_init(__global struct allocator *a, __global void *buffer,
			     uint32_t size, uint32_t chunk_size, int flags)
{
	uint32_t chunks, chunk_shift, nbytes_bitmap, converge = 0;

	if (!buffer)
		return -EINVAL;
	if (!is_power_of_two(size) || !is_power_of_two(chunk_size))
		return -EINVAL;
	if (chunk_size >= size)
		return -EINVAL;
	if (chunk_size <= sizeof(*a->free_bitmap))
		return -EINVAL;

	chunk_shift = ilog2(chunk_size);
	chunks = size >> chunk_shift;
	while (1) {
		uint32_t real_chunks;

		nbytes_bitmap = round_up(chunks, sizeof(*a->free_bitmap) << 3) >> 3;
		real_chunks = (size - nbytes_bitmap) >> chunk_shift;
		if (chunks != real_chunks && converge < 2) {
			/* Recalculate once more taking bitmap into account */
			converge += (abs(chunks - real_chunks) == 1);
			chunks = real_chunks;
			continue;
		}
		break;
	}

	a->buffer = buffer;
	/* Make OpenCL gcc happy */
	a->free_bitmap = (__global uint64_t*)((__global char *)buffer +	(chunks << chunk_shift));
	a->nbytes_bitmap = nbytes_bitmap;
	a->nlongs_bitmap = nbytes_bitmap / sizeof(*a->free_bitmap);
	a->chunk_shift = chunk_shift;
	a->chunk_size = chunk_size;
	a->chunks = chunks;
	a->size = size;
	a->free_chunks = chunks;
	a->fail_on_nomem = (flags & FAIL_ON_NOMEM);

	init_free_bitmap(a);

	return 0;
}

static inline int alloc_deinit(__global struct allocator *a)
{
	__global uint8_t *bytes = (__global uint8_t *)a->free_bitmap;
	uint32_t first_half = a->chunks >> 3;

	/* Check everything was freed */
	if (a->free_chunks != a->chunks)
		return -EINVAL;

	/* Check free bitmap is not corrupted */
	if (first_half &&
	    (bytes[0] != 0xff || memcmp(bytes, bytes + 1, first_half - 1))) {
		return -EINVAL;
	}
	if ((a->chunks & 7) && bytes[first_half] != (1<<(a->chunks % 8)) - 1)
		return -EINVAL;

	return 0;
}

static inline __global void *__alloc_chunk(__global struct allocator *a, uint32_t i)
{
	__global uint64_t *p;
	uint64_t old, new;
	uint32_t bit, ch;

	p = &a->free_bitmap[i];
	do {
		old = *p;
		bit = ffs_bit(old);
		if (!bit)
			return NULL;
		new = old & ~(1ul<<(bit-1));
	} while (atomic64_cmpxchg(p, old, new) != old);

	atomic32_dec(&a->free_chunks);
	ch = (i << (ilog2(sizeof(*a->free_bitmap)) + 3)) + (bit-1);

	/* Make OpenCL gcc happy */
	return (__global char *)a->buffer + (ch << a->chunk_shift);
}

static inline int __free_chunk(__global struct allocator *a, uint32_t chunk)
{
	__global uint64_t *p;
	uint64_t old, new;
	uint32_t bit, i, shift;

	shift = ilog2(sizeof(*a->free_bitmap)) + 3;
	bit = chunk & ((1ul<<shift)-1);
	i = chunk >> shift;

	p = &a->free_bitmap[i];
	do {
		old = *p;
		new = old | (1ul<<bit);
		if (old == new) {
			/* Double free? */
			return -EINVAL;
		}
	} while (atomic64_cmpxchg(p, old, new) != old);

	atomic32_inc(&a->free_chunks);

	return 0;
}

/**
 * alloc_chunk() - allocates fixed-size chunk. @hint can be specified
 * in order to reduce contention on the same cache line if many threads
 * heavily use allocator.
 */
static inline __global void *alloc_chunk(__global struct allocator *a, int hint)
{
	uint32_t i, start = hint % a->nlongs_bitmap;
	__global void *chunk;

	do {
		/* First half */
		for (i = start; i < a->nlongs_bitmap; i++) {
			chunk = __alloc_chunk(a, i);
			if (chunk)
				return chunk;
		}
		/* Second half */
		for (i = 0; i < start; i++) {
			chunk = __alloc_chunk(a, i);
			if (chunk)
				return chunk;
		}
	} while (!a->fail_on_nomem);

	return NULL;
}

static inline int free_chunk(__global struct allocator *a, __global void *chunk)
{
	uintptr_t ch;

	if (!chunk)
		return 0;

	if (chunk < a->buffer)
		/* Beyond? Invalid */
		return -EINVAL;

	ch = chunk - a->buffer;
	if (ch & ((1 << a->chunk_shift)-1))
		/* Not aligned? Invalid */
		return -EINVAL;

	ch >>= a->chunk_shift;
	if (ch >= a->chunks)
		/* Beyound? Invalid */
		return -EINVAL;

	return __free_chunk(a, ch);
}

#endif /* ALLOC_H */
