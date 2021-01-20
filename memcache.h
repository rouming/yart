/**
 * Simple fixed-size memcache allocator. Utilizes alloc.h as
 * an allocator for big fixed-size chunks, where each chunks
 * is divided on @cache_size blocks. Allocator is simple and
 * is not thread-safe, should be used small local allocations.
 *
 * 2021, Roman Penyaev <r.peniaev@gmail.com>
 */

#ifndef MEMCACHE_H
#define MEMCACHE_H

#include "alloc.h"

struct memcache_chunk {
	uint32_t        nr_free;
	uint32_t        free_bitmap[];
};

struct memcache {
	__global struct allocator *a;
	__global struct memcache_chunk
			 *cached_chunk;
	uint16_t         cache_size;
	uint16_t         blocks;
	uint16_t         nbytes_bitmap;
	uint16_t         alloced;
};

static inline int
memcache_init(struct memcache *mc, __global struct allocator *a,
	      uint16_t cache_size)
{
	uint32_t size, blocks, nbytes_bitmap;

	if (!cache_size)
		return -EINVAL;
	if ((cache_size + sizeof(struct memcache_chunk)) >= a->chunk_size)
		return -EINVAL;

	size = a->chunk_size - sizeof(struct memcache_chunk);

	blocks = size / cache_size;
	nbytes_bitmap = (round_up(blocks, 32) >> 3);
	blocks = (size - nbytes_bitmap) / cache_size;
	if (!blocks)
		/* Not enough */
		return -EINVAL;

	mc->blocks = blocks;
	mc->nbytes_bitmap = nbytes_bitmap;

	mc->a = a;
	mc->cache_size = cache_size;
	mc->alloced = 0;
	mc->cached_chunk = NULL;

	return 0;
}

static inline int memcache_deinit(struct memcache *mc)
{
	if (mc->cached_chunk)
		free_chunk(mc->a, mc->cached_chunk);

	/* Nothing we can do here, just report an error */
	return mc->alloced ? -EINVAL : 0;
}

static inline void
memcache_init_free_bitmap(struct memcache *mc,
			  __global struct memcache_chunk *chunk)
{
	uint32_t first_half = mc->blocks >> 3;
	uint32_t last_half = mc->nbytes_bitmap - (round_up(mc->blocks, 8) >> 3);
	__global uint8_t *bytes = (__global uint8_t *)chunk->free_bitmap;

	chunk->nr_free = mc->blocks;

	if (first_half)
		memset(bytes, 0xff, first_half);
	if (mc->blocks & 7)
		bytes[first_half] = (1<<(mc->blocks & 7)) - 1;
	if (last_half)
		memset(bytes + first_half + 1, 0, last_half);
}

static inline __global void *memcache_alloc(struct memcache *mc)
{
	__global struct memcache_chunk *chunk;
	uint32_t i, bit;

	if (!(chunk = mc->cached_chunk)) {
		chunk = alloc_chunk(mc->a, get_alloc_hint());
		mc->cached_chunk = chunk;
		if (!chunk)
			/* No memory */
			return NULL;

		memcache_init_free_bitmap(mc, chunk);
	}

	for (bit = 0, i = 0; i < mc->nbytes_bitmap >> 2; i++) {
		uint32_t v = chunk->free_bitmap[i];

		bit = ffs_bit(v);
		if (!bit)
			continue;

		chunk->free_bitmap[i] = v & ~(1<<(bit-1));
		break;
	}
	if (!bit || !chunk->nr_free) {
		/* Should not happen */
		return NULL;
	}
	mc->alloced++;
	if (!--chunk->nr_free)
		mc->cached_chunk = NULL;

	return (__global char *)chunk->free_bitmap + mc->nbytes_bitmap +
		(mc->cache_size * ((i<<5) + (bit-1)));
}

static inline int memcache_free(struct memcache *mc, __global void *p)
{
	__global struct memcache_chunk *chunk;
	__global void *buf;
	uint32_t ch, shift, bit, i;
	uintptr_t off;

	if (!p)
		return 0;

	chunk = ALIGN_PTR_DOWN(p, (uintptr_t)mc->a->chunk_size);
	if (chunk == p)
		/* Invalid? */
		return -EINVAL;

	buf = (__global void *)chunk->free_bitmap + mc->nbytes_bitmap;
	if (p < buf)
		/* Invalid? */
		return -EINVAL;

	off = p - buf;
	if (off % mc->cache_size)
		/* Invalid? */
		return -EINVAL;

	ch = off / mc->cache_size;

	shift = 32;
	bit = ch & ((1<<shift)-1);
	i = ch >> shift;

	chunk->free_bitmap[i] |= (1<<bit);
	chunk->nr_free++;
	mc->alloced--;

	if (!mc->cached_chunk || chunk->nr_free > mc->cached_chunk->nr_free) {
		mc->cached_chunk = chunk;
	} else if (chunk != mc->cached_chunk && chunk->nr_free >= mc->blocks) {
		(void)free_chunk(mc->a, chunk);
	}

	return 0;
}

#endif /* MEMCACHE_H */
