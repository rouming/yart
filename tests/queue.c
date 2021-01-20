#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define __global

#include "../bvh.h"

static int float_ascending(const void *a, const void *b)
{
	const float *aa = a;
	const float *bb = b;

	return *aa - *bb;
}

int main(int argc, char **argv)
{
	struct allocator alloc;
	struct octant_queue queue;

	uint32_t size, chunk_size;
	int ret, i, nr_elems;
	void *p;
	float *rands;

	if (argc != 3) {
		printf("Usage: <size> <chunk_size>\n");
		return -1;
	}

	size = atoi(argv[1]);
	chunk_size = atoi(argv[2]);

	srand(0);

	p = malloc(size);
	assert(p);
	/* Poison */
	memset(p, 0xca, size);

	ret = alloc_init(&alloc, p, size, chunk_size, FAIL_ON_NOMEM);
	assert(!ret);

	ret = octant_queue_init(&queue, &alloc);
	assert(!ret);

	/* Roughly */
	nr_elems = size / queue.mc.cache_size;

	printf(">> roughly nr_elems=%d\n", nr_elems);

	rands = malloc(sizeof(*rands) * nr_elems);
	assert(rands);

	for (i = 0; i < nr_elems; i++) {
		void *addr;

		if (i < 4) {
			/* Insert several equal floats */
			rands[i] = 1.0;
			addr = (void *)(uintptr_t)(i + 1);
		} else {
			/* Except zero */
			while (!(rands[i] = (float)rand()))
				;
			addr = (void *)(uintptr_t)rands[i];
		}

		ret = octant_queue_insert(&queue, addr, rands[i]);
		if (ret == -ENOMEM) {
			/* Ok, get exact number of elements */
			printf(">> nr_elems=%d, was %d\n", i, nr_elems);
			nr_elems = i;
			break;
		}
		assert(!ret);
	}

	/* Ascending order */
	qsort(rands, nr_elems, sizeof(*rands), float_ascending);

	for (i = 0; i < nr_elems; i++) {
		void *addr;
		float f1, f2 = 0.0;

		addr = octant_queue_pop_first(&queue, &f2);
		assert(addr);

		f1 = (float)(uintptr_t)addr;
		if (f2 != 1.0)
			assert(f1 == f2);
		assert(rands[i] == f2);
	}


	ret = octant_queue_deinit(&queue);
	assert(!ret);

	ret = alloc_deinit(&alloc);
	assert(!ret);

	return 0;
}
