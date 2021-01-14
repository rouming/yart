/**
 * Simple index-tree array implementation. Array consists of nodes,
 * contigous fixed-size memory chunks (see alloc.h). Leaf nodes have
 * array elements (elemet size is specified on array_init() call),
 * branch nodes consist of pointers pointing on further branch nodes
 * (when height > 1) or to leaf nodes (height == 1). When array is
 * created it consists of a single leaf node, i.e.  height == 0. When
 * array is grow (array_push() is called) and all elements are
 * occupied, new root node is created and all children are reasigned
 * to a new root. When array is shrinked (array_pop() is called)
 * ->i_begin offset is increased to have a bias for future array_get()
 * or array_set() operations.  When ->i_begin becomes equal to
 * array_capacity_below() the whole branch is freed, pointers are
 * moved for root node and ->i_begin is set to 0.
 *
 * 2021, Roman Penyaev <r.peniaev@gmail.com>
 */

#ifndef ARRAY_H
#define ARRAY_H

#ifndef __OPENCL__
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#endif

#include "alloc.h"

struct array {
	__global void    *node;
	__global struct allocator
			  *a;
	uint32_t          i_begin;            /* index offset to perform fast pop */
	uint32_t          nr_elems;           /* number of all elements in the array */
	uint16_t          height;             /* height of the whole array tree */
	uint16_t          elem_size;          /* size of an element in a leaf node */
	uint16_t          leaf_elems;         /* number of elements in a leaf node */
	uint16_t          branch_elems;       /* number of pointers in a node */
	uint16_t          branch_elems_shift; /* shift of pointers in a node */
};

static inline uint32_t __array_capacity(struct array *array, uint32_t height)
{
	return array->leaf_elems * (1<<(array->branch_elems_shift * height));
}

static inline uint32_t array_capacity(struct array *array)
{
	return __array_capacity(array, array->height);
}

static inline uint32_t array_capacity_below(struct array *array, uint32_t height)
{
	return height ? __array_capacity(array, height - 1) : 0;
}

static inline int array_init(struct array *array, __global struct allocator *a,
			     uint32_t elem_size)
{
	uint32_t branch_elem_shift, branch_elems_shift;
	uint32_t leaf_elems, branch_elems;

	branch_elem_shift = ilog2(sizeof(void*));

	if (elem_size >= a->chunk_size)
		return -EINVAL;
	if (a->chunk_shift <= branch_elem_shift)
		return -EINVAL;

	array->node = alloc_chunk(a, get_alloc_hint());
	if (!array->node)
		return -ENOMEM;

	leaf_elems = a->chunk_size / elem_size;
	branch_elems_shift = a->chunk_shift - branch_elem_shift;
	branch_elems = 1<<branch_elems_shift;

	array->a = a;
	array->height = 0; /* 0 is a leaf node */
	array->elem_size = elem_size;
	array->leaf_elems = leaf_elems;
	array->branch_elems = branch_elems;
	array->branch_elems_shift = branch_elems_shift;
	array->i_begin = 0;
	array->nr_elems = 0;

	return 0;
}

static inline int
__array_free_from_node(struct array *array, __global void *root_node,
		       uint32_t root_height)
{
	uint32_t nr_elems = array->nr_elems + array->i_begin;

	/* Can't use recursion, so go deep to the left and repeat from top */
	while (root_height) {
		uint32_t height = root_height;
		__global void *node = root_node;
		__global void **slot = NULL;
		uint32_t nr;
		int i, ret;

		/* Go deep to the left */
		for (nr = 0; height; height--) {
			__global void * __global *pptr = node;
			uint32_t capacity;

			capacity = array_capacity_below(array, height);
			for (i = 0; nr < nr_elems && i < array->branch_elems;
			     i++, nr += capacity) {
				if (!pptr[i])
					/* Already freed */
					continue;
				if (height > 1) {
					/* Continue descent */
					slot = &pptr[i];
					node = pptr[i];
					goto deeper;
				}
				/* Free leaves */
				ret = free_chunk(array->a, pptr[i]);
				if (ret)
					return ret;
				pptr[i] = NULL;
			}
			if (slot) {
				assert(*slot);
				ret = free_chunk(array->a, *slot);
				if (ret)
					return ret;
				*slot = NULL;
				slot = NULL;
				/* Repeat from top */
				break;
			} else if (height == root_height) {
				/*
				 * Did not descent because all children node are freed,
				 * thus we are done.
				 */
				goto done;
			}
deeper:
			/* Make gcc happy */
			continue;
		}
	}
done:
	return 0;
}

static inline int array_deinit(struct array *array)
{
	int ret;

	ret = __array_free_from_node(array, array->node, array->height);
	if (ret)
		return ret;

	/* Free last root node */
	return free_chunk(array->a, array->node);
}

static inline __global void *array_get(struct array *array, uint32_t i)
{
	uint32_t height;
	__global void *node;

	if (i >= array->nr_elems)
		return NULL;

	i += array->i_begin;

	height = array->height;
	node = array->node;
	for ( ; height && node; height--) {
		uint32_t i_node, capacity;
		__global void * __global *pptr = node;

		capacity = array_capacity_below(array, height);
		i_node = i / capacity;
		i -= i_node * capacity;
		node = pptr[i_node];
	}
	if (!node)
		return NULL;

	/* Make OpenCL gcc happy */
	return (__global char *)node + i * array->elem_size;
}

static inline int array_set(struct array *array, uint32_t i, void *p)
{
	__global void *elem;

	elem = array_get(array, i);
	if (!elem)
		return -ENOENT;
	memcpy_to_global(elem, p, array->elem_size);

	return 0;
}

static inline int array_push_tail(struct array *array, void *p)
{
	uint32_t i, height, capacity, prior_capacity;
	__global void *node;

	i = array->nr_elems + array->i_begin;

	capacity = array_capacity(array);
	assert(i <= capacity);

	if (i == capacity) {
		/* Need to increase tree height */
		__global void * __global *pptr;

		pptr = alloc_chunk(array->a, get_alloc_hint());
		if (!pptr)
			return -ENOMEM;

		pptr[0] = array->node;
		array->node = pptr;
		array->height++;
	}

	height = array->height;
	node = array->node;
	prior_capacity = 0;
	for ( ; height; height--) {
		uint32_t i_node, capacity;
		__global void * __global *pptr = node;

		capacity = array_capacity_below(array, height);
		i_node = i / capacity;
		prior_capacity += i_node * capacity;
		i -= i_node * capacity;

		if (prior_capacity == array->nr_elems) {
			/* Create new child node */
			node = alloc_chunk(array->a, get_alloc_hint());
			if (!node)
				return -ENOMEM;

			pptr[i_node] = node;
		}
		node = pptr[i_node];
	}
	/* Make OpenCL gcc happy */
	memcpy_to_global((__global char *)node + i * array->elem_size,
			 p, array->elem_size);

	array->nr_elems++;

	return 0;
}

static inline int array_pop_head(struct array *array, void *p)
{
	__global void *elem;
	uint32_t capacity;

	if (!array->nr_elems)
		return -ENOENT;

	elem = array_get(array, 0);
	if (!elem)
		/* Hm, actually unreachable line */
		return -ENOENT;

	memcpy_from_global(p, elem, array->elem_size);
	array->nr_elems--;
	array->i_begin++;

	capacity = array_capacity_below(array, array->height);
	if (array->height && array->i_begin >= capacity) {
		/* Can free the whole branch */
		__global void * __global *pptr = array->node;
		__global void *node = pptr[0];
		int ret;

		assert(node);
		ret = __array_free_from_node(array, node, array->height - 1);
		/* Not very much we can do in case of error */
		assert(!ret);

		ret = free_chunk(array->a, node);
		/* Not very much we can do in case of error */
		assert(!ret);

		/*
		 * Move all pointers for the root node to the left
		 * one one and drop i_begin offset.
		 */
		memmove(array->node,
			/* Make OpenCL gcc happy */
			(__global char *)array->node + sizeof(void*),
			array->a->chunk_size - sizeof(void*));
		array->i_begin = 0;
	}

	return 0;
}

#endif /* ARRAY_H */
