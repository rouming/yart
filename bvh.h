#ifndef BVH_H
#define BVH_H

#include "types.h"
#include "math_3d.h"
#include "list.h"

enum {
	NR_PLANE_NORMALS = 7,
};

static __constant const vec3_t plane_set_normals[] = {
	[0] = {1.0f, 0.0f, 0.0f},
	[1] = {0.0f, 1.0f, 0.0f},
	[2] = {0.0f, 0.0f, 1.0f},
	/* sqrt(3)/3.0f */
	[3] = { 0.577350,  0.577350, 0.577350},
	[4] = {-0.577350,  0.577350, 0.577350},
	[5] = {-0.577350, -0.577350, 0.577350},
	[6] = { 0.577350, -0.577350, 0.577350},
};

struct slab {
	float near;
	float far;
};

struct extent {
	/**
	 * 'd' components for near and far plane-sets of plane equation:
	 *    Ax + By + Cz - d = 0
	 *    N.P - d = 0 ; where P = O + t*Dir
	 *    N.(O + t*Dir) = d
	 *    N.O + t*N.Dir = d
	 *    t = (d - N.O) / (N.Dir); where 't' is a distance
	 */
	struct slab d[NR_PLANE_NORMALS];
};

struct extent_leaf {
	struct extent       e;
	struct list_head    entry; /* entry in octant->leaves */
	__global struct triangle_mesh
			    *mesh;
	uint32_t            index;  /* index of first triangle vertex in mesh */
};

enum {
	OCTANT_CACHE_SIZE = 1024,

	OCTANT_MAX_LEAVES = 128,
	OCTANT_MAX_DEPTH  = 32,
};

struct octant {
	union {
		__global struct octant *octants[8];
		struct list_head       entry; /* entry in bvh->stat.octants_leaves */
	};
	struct list_head       leaves;       /* list of extent_leaf */
	struct extent          extent;
	uint32_t               __leaves_cnt; /* see octant_is/set_leaf() */
};

struct octant_cache {
	__global struct octant *octants;
};

enum {
	LESS_10  = 0,
	LESS_100,
	LESS_1000,
	LESS_10000,
	UPPER_BOUND,
};

struct bvhtree_stat {
	uint32_t                    max_depth;
	uint32_t                    max_leaves;
	uint32_t                    num_octants[UPPER_BOUND+1];
	struct list_head            octants_leaves;
};

struct bvhtree {
	struct octant               root;
	__global struct extent_leaf *leaves;
	struct octant_cache         *octant_caches;
	struct bvhtree_stat         stat;
	uint32_t                    num_caches;
	uint32_t                    num_octants;
};

struct scene;

/* BVH host public API */
void bvhtree_init(struct bvhtree *bvh);
void bvhtree_deinit(struct bvhtree *bvh);
int bvhtree_build(struct bvhtree *bvh, struct scene *scene);

/*
 * The following API can be called from any CPU and GPU
 */

__accelerated static inline bool octant_is_leaf(const __global struct octant *octant)
{
	return octant->__leaves_cnt;
}

__accelerated static inline bool extent_intersect(const __global struct extent *e,
				    float numerators[NR_PLANE_NORMALS],
				    float denominators[NR_PLANE_NORMALS],
				    float *t_near, float *t_far)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(e->d); i++) {
		float t_near_ext = (e->d[i].near - numerators[i]) / denominators[i];
		float t_far_ext  = (e->d[i].far - numerators[i]) / denominators[i];

		if (denominators[i] < 0.0f) {
			SWAP(t_near_ext, t_far_ext);
		}
		if (t_near_ext > *t_near) {
			*t_near = t_near_ext;
		}
		if (t_far_ext < *t_far) {
			*t_far = t_far_ext;
		}
		if (*t_near > *t_far) {
			return false;
		}
	}

	return true;
}

struct octant_queue_entry {
	__global struct octant *octant;
	float                   t;
	uint32_t                _pad;
};

struct octant_queue {
	__global struct octant_queue_entry *entries;
	uint32_t count;
	uint32_t max;
};

__accelerated static inline void octant_queue_init(struct octant_queue *queue,
				    __global struct octant_queue_entry *entries,
				    uint32_t max)
{
	queue->entries = entries;
	queue->count = 0;
	queue->max = max;
}

__accelerated static inline void octant_queue_deinit(struct octant_queue *queue)
{
	(void)queue;
}

__accelerated static inline __global struct octant *
octant_queue_pop_first(struct octant_queue *queue, float *t)
{
	__global struct octant *octant;
	uint32_t i;

	if (!queue->count)
		return NULL;

	octant = queue->entries[0].octant;
	*t = queue->entries[0].t;

	queue->count--;
	for (i = 0; i < queue->count; i++)
		queue->entries[i] = queue->entries[i + 1];

	return octant;
}

__accelerated static inline int octant_queue_insert(struct octant_queue *queue,
				      __global struct octant *octant,
				      float t)
{
	uint32_t i, pos;

	if (queue->count >= queue->max) {
		assert(0);
		return -ENOMEM;
	}

	/*
	 * Find insertion position: sorted ascending by t;
	 * equal t goes after existing entries (FIFO order).
	 */
	for (pos = 0; pos < queue->count && queue->entries[pos].t <= t; pos++)
		;

	/* Shift entries up to make room */
	for (i = queue->count; i > pos; i--)
		queue->entries[i] = queue->entries[i - 1];

	queue->entries[pos].octant = octant;
	queue->entries[pos].t = t;
	queue->count++;

	return 0;
}

#endif /* BVH_H */
