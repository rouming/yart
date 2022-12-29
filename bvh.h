#ifndef BVH_H
#define BVH_H

#include "types.h"
#include "math_3d.h"
#include "memcache.h"
#include "list.h"
#include "rbtree.h"

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
	__global struct allocator   *alloc;
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

static inline bool octant_is_leaf(const __global struct octant *octant)
{
	return octant->__leaves_cnt;
}

static inline bool extent_intersect(const __global struct extent *e,
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

struct octant_elem {
	struct rb_node entry;   /* entry in octant_queue->root */
	__global struct octant *octant;
	float                   t;
};

struct octant_queue {
	struct memcache mc;
	__global struct rb_root  *root;
};

static inline int octant_queue_init(struct octant_queue *queue,
				    __global struct allocator *a)
{
	int ret;

	ret = memcache_init(&queue->mc, a, sizeof(struct octant_elem));
	if (ret)
		return ret;

	queue->root = NULL;

	return 0;
}

static inline int octant_queue_deinit(struct octant_queue *queue)
{
	__global struct octant_elem *elem;
	__global struct rb_node *node;

	if (!queue->root)
		return 0;

	/* Free queue node by node */
	/*
	 * Optimization hint: don't free node by node, but simply
	 * free chunks, we don't care about proper queue tree elements
	 * destruction, we care only about freeing allocator chunks.
	 */
	while ((node = rb_first(queue->root)) != NULL) {
		rb_erase(node, queue->root);
		elem = container_of(node, typeof(*elem), entry);
		(void)memcache_free(&queue->mc, elem);
	}
	(void)memcache_free(&queue->mc, queue->root);

	return memcache_deinit(&queue->mc);
}

static inline __global struct octant *
octant_queue_pop_first(struct octant_queue *queue, float *t)
{
	__global struct octant_elem *elem;
	__global struct octant *octant;
	__global struct rb_node *node;

	if (!queue->root || RB_EMPTY_ROOT(queue->root))
		return NULL;

	node = rb_first(queue->root);
	elem = container_of(node, typeof(*elem), entry);
	*t = elem->t;

	rb_erase(node, queue->root);
	octant = elem->octant;
	memcache_free(&queue->mc, elem);

	return octant;
}

static float cmp_octants(__global const struct rb_node *a_,
			 __global const struct rb_node *b_)
{
	__global struct octant_elem *a;
	__global struct octant_elem *b;

	a = container_of(a_, typeof(*a), entry);
	b = container_of(b_, typeof(*b), entry);

	return a->t - b->t;
}

static inline int octant_queue_insert(struct octant_queue *queue,
				      __global struct octant *octant,
				      float t)
{
	__global struct rb_node * __global * this;
	__global struct rb_node *parent = NULL;
	__global struct octant_elem *elem;
	__global struct rb_node *new;
	float cmp;

	if (!queue->root) {
		queue->root = memcache_alloc(&queue->mc);
		if (!queue->root)
			return -ENOMEM;

		*queue->root = RB_ROOT;
	}
	this = &queue->root->rb_node;

	elem = memcache_alloc(&queue->mc);
	if (!elem)
		return -ENOMEM;

	elem->octant = octant;
	elem->t = t;

	new = &elem->entry;
	while (*this) {
		parent = *this;
		cmp = cmp_octants(new, *this);

		/*
		 * Equal elements go to the right, i.e. FIFO order, thus '<',
		 * if LIFO is needed then use '<='
		 */
		if (cmp < 0.0f)
			this = &(*this)->rb_left;
		else
			this = &(*this)->rb_right;
	}
	/* Add new node to the tree and rebalance it */
	rb_link_node(new, parent, this);
	rb_insert_color(new, queue->root);

	return 0;
}

#endif /* BVH_H */
