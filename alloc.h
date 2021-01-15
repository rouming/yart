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

#include "types.h"

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
