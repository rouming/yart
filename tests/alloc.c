#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define __global

#include "../alloc.h"

int main(int argc, char **argv)
{
	struct allocator alloc;
	uint32_t size, chunk_size;
	void **chunks;
	uint32_t *rands;
	void *chunk, *p;
	int i, ret;

	if (argc != 3) {
		printf("Usage: <size> <chunk_size>\n");
		return -1;
	}

	size = atoi(argv[1]);
	chunk_size = atoi(argv[2]);

	p = malloc(size);
	/* Poison */
	memset(p, 0xca, size);

	ret = alloc_init(&alloc, p, size, chunk_size, FAIL_ON_NOMEM);
	assert(!ret);

	srand(0);

	chunks = malloc(sizeof(*chunks) * alloc.free_chunks);
	assert(chunks);
	rands = malloc(sizeof(*rands) * alloc.free_chunks);
	assert(rands);

	i = 0;
	do {
		int hint = rand();
		chunk = alloc_chunk(&alloc, hint);
		if (chunk) {
			uint32_t *v = chunk;
			int ii;

			for (ii = 0; ii < i; ii++) {
				if (chunks[ii] == chunk) {
					printf("chunk=%p ii=%d, i=%d\n", chunk, ii, i);
				}
				assert(chunks[ii] != chunk);
			}

			for (ii = 0; ii < chunk_size / 4; ii++)
				v[ii] = hint;


			rands[i] = hint;
			chunks[i++] = chunk;
		}
	} while (chunk);

	assert(alloc.free_chunks == 0);
	assert(alloc.chunks == i);

	while (i > 0) {
		uint32_t *v = chunks[--i];
		int ii;
		for (ii = 0; ii < chunk_size / 4; ii++) {
			assert(v[ii] == rands[i]);
		}
		ret = free_chunk(&alloc, chunks[i]);
		if (ret)
			printf("?? i=%d\n", i+1);
		assert(!ret);
	}
	ret = alloc_deinit(&alloc);
	assert(!ret);

	return 0;
}
