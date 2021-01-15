#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define __global

#include "../array.h"

int main(int argc, char **argv)
{
	struct allocator alloc;
	struct array arr;

	uint32_t size, chunk_size, elem_size;
	int ret, i, ii, nr_elems;
	void *p, *elem, **elems;
	uint32_t *rands;

	size       = 1<<30;
	chunk_size = 1<<10;
	elem_size  = 1<<3;
	nr_elems   = 1<<20;

	srand(0);

	p = malloc(size);
	assert(p);
	/* Poison */
	memset(p, 0xca, size);

	ret = alloc_init(&alloc, p, size, chunk_size, FAIL_ON_NOMEM);
	assert(!ret);

	elems = malloc(sizeof(*elems) * nr_elems);
	assert(elems);
	rands = malloc(sizeof(*rands) * nr_elems);
	assert(rands);

	for (i = 0; i < nr_elems; i++) {
		elem = malloc(elem_size);
		assert(elem);
		rands[i] = rand();
		for (ii = 0; ii < elem_size / 4; ii++)
			((uint32_t *)elem)[ii] = rands[i];
		elems[i] = elem;
	}

	ret = array_init(&arr, &alloc, elem_size);
	assert(!ret);

	for (i = 0; i < nr_elems; i++) {
		elem = array_push_tail(&arr);
		if (!elem) {
			printf("Allocation failed.\n");
		}
		assert(elem);
		memcpy(elem, elems[i], elem_size);
	}

	for (i = 0; i < nr_elems; i++) {
		elem = array_get(&arr, i);
		assert(elem);

		for (ii = 0; ii < elem_size / 4; ii++) {
			uint32_t v = ((uint32_t *)elem)[ii];
			if (v != rands[i]) {
				printf("?? ii=%d[%d] i=%d[%d]\n",
				       ii, v, i, rands[i]);
			}
			assert(v == rands[i]);
		}
		assert(!memcmp(elems[i], elem, elem_size));
	}
	printf(">>    nr elems %d\n", arr.nr_elems);
	printf(">>      height %d\n", arr.height);
	printf(">>   nr leaves %d\n", arr.leaf_elems);
	printf(">>   nr branch %d\n", arr.branch_elems);
	printf(">> free chunks %d\n", alloc.free_chunks);

	elem = malloc(elem_size);
	assert(elem);

	for (i = 0; i < nr_elems/2; i++) {
		elem = array_get(&arr, 0);
		assert(elem);
		ret = array_pop_head(&arr);
		assert(!ret);

		assert(!memcmp(elems[i], elem, elem_size));
	}
	for (i = nr_elems/2; i < nr_elems; i++) {
		elem = array_get(&arr, i - nr_elems/2);
		assert(elem);

		assert(!memcmp(elems[i], elem, elem_size));
	}

	ret = array_deinit(&arr);
	assert(!ret);

	ret = alloc_deinit(&alloc);
	assert(!ret);

	return 0;
}
