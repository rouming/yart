#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define __global

#include "../memcache.h"

int main(int argc, char **argv)
{
	struct allocator alloc;
	struct memcache cache;

	uint32_t size, chunk_size, cache_size;
	int ret, i, ii, nr_elems;
	void *p, *elem, **elems;
	uint32_t *rands;

	if (argc != 4) {
		printf("Usage: <size> <chunk_size> <cache_size>\n");
		return -1;
	}

	size = atoi(argv[1]);
	chunk_size = atoi(argv[2]);
	cache_size = atoi(argv[3]);
	/* Rough but should work */
	nr_elems = size / cache_size;

	printf(">> roughly nr_elems=%d\n", nr_elems);

	p = malloc(size);
	/* Poison */
	memset(p, 0xca, size);

	ret = alloc_init(&alloc, p, size, chunk_size, FAIL_ON_NOMEM);
	assert(!ret);

	ret = memcache_init(&cache, &alloc, cache_size);
	assert(!ret);

	srand(0);

	elems = malloc(sizeof(*elems) * nr_elems);
	assert(elems);
	rands = malloc(sizeof(*rands) * nr_elems);
	assert(rands);

	for (i = 0; i < nr_elems; i++) {
		elem = memcache_alloc(&cache);
		if (!elem) {
			/* Ok, get exact number of elements */
			printf(">> nr_elems=%d, was %d\n", i, nr_elems);
			nr_elems = i;
			break;
		}

		for (ii = 0; ii < i; ii++) {
			if (elems[ii] == elem) {
				printf("elem=%p ii=%d, i=%d\n", elem, ii, i);
			}
			assert(elems[ii] != elem);
		}
		elems[i] = elem;
		rands[i] = rand();
		for (ii = 0; ii < cache_size / 4; ii++)
			((uint32_t *)elem)[ii] = rands[i];
	}

	while (i > 0) {
		uint32_t *v = elems[--i];
		int ii;
		for (ii = 0; ii < cache_size / 4; ii++) {
			assert(v[ii] == rands[i]);
		}
		ret = memcache_free(&cache, elems[i]);
		if (ret)
			printf("?? i=%d, ret=%d\n", i+1, ret);
		assert(!ret);
	}
	ret = memcache_deinit(&cache);
	assert(!ret);

	ret = alloc_deinit(&alloc);
	assert(!ret);

	return 0;
}
