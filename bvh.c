#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "scene.h"
#include "bvh.h"
#include "buf.h"

static void extent_init(struct extent *e)
{
	int i;

	for (i = 0;  i < ARRAY_SIZE(e->d); i++) {
                e->d[i].near =  INFINITY;
		e->d[i].far  = -INFINITY;
	}
}

static void extent_expand(struct extent *this, const struct extent *that)
{
	int i;

	for (i = 0;  i < ARRAY_SIZE(this->d); i++) {
		this->d[i].near = MIN(this->d[i].near, that->d[i].near - EPSILON);
		this->d[i].far  = MAX(this->d[i].far,  that->d[i].far  + EPSILON);
	}
}

static void extent_expand_by_vertex(struct extent *e, vec3_t v)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(e->d); i++) {
		float d = v3_dot(v, plane_set_normals[i]);

		e->d[i].near = MIN(e->d[i].near, d - EPSILON);
		e->d[i].far  = MAX(e->d[i].far,  d + EPSILON);
	}
}

static vec3_t extent_centroid(const struct extent *e)
{
	return vec3((e->d[0].near + e->d[0].far) * 0.5,
		    (e->d[1].near + e->d[1].far) * 0.5,
		    (e->d[2].near + e->d[2].far) * 0.5);
}

static void extent_leaf_init(struct extent_leaf *ext,
			     struct triangle_mesh *mesh,
			     int index)
{
	int i, n;

	ext->mesh = mesh;
	ext->index = index;
	extent_init(&ext->e);

	/* Expand by each vertex in triangle */
	for (i = index, n = i + 3; i < n; i++)
		extent_expand_by_vertex(&ext->e, mesh->vertices[i]);
}

static inline void octant_set_leaf(struct octant *octant, bool is_leaf)
{
	octant->__leaves_cnt = !!is_leaf;
	if (!is_leaf) {
		list_del(&octant->entry);
		memset(octant->octants, 0, sizeof(octant->octants));
	}
}

static inline uint32_t octant_num_leaves(const struct octant *octant)
{
	if (octant_is_leaf(octant))
		return octant->__leaves_cnt - 1;
	return 0;
}

static inline void octant_inc_leaves(struct octant *octant)
{
	octant->__leaves_cnt++;
}

static void octant_init(struct bvhtree *bvh, struct octant *octant)
{
	list_add_tail(&octant->entry, &bvh->stat.octants_leaves);
	INIT_LIST_HEAD(&octant->leaves);
	extent_init(&octant->extent);
	octant_set_leaf(octant, true);
}

static void bvhtree_init_stat(struct bvhtree *bvh)
{
	struct bvhtree_stat *stat = &bvh->stat;

	INIT_LIST_HEAD(&bvh->stat.octants_leaves);
	memset(stat->num_octants, 0, sizeof(stat->num_octants));
	stat->max_depth  = 0;
	stat->max_leaves = 0;
}

static void bvhtree_update_stat(struct bvhtree *bvh, struct octant *octant,
				int depth)
{
	struct bvhtree_stat *stat = &bvh->stat;

	stat->max_depth  = MAX(stat->max_depth,  depth);
	stat->max_leaves = MAX(stat->max_leaves, octant_num_leaves(octant));
}

__attribute__((unused))
static void bvhtree_print_stat(struct bvhtree *bvh)
{
	struct bvhtree_stat *stat = &bvh->stat;
	int i, n;

	printf(">> max_depth=%d\n", bvh->stat.max_depth);
	printf(">> max_leaves=%d\n", bvh->stat.max_leaves);
	for (i = 0, n = 10; i <= UPPER_BOUND; i++, n *= 10) {
		if (i < UPPER_BOUND)
			printf("  <  %6d: %d\n", n, stat->num_octants[i]);
		else
			printf("  >= %6d: %d\n", n/10, stat->num_octants[i]);
	}
}

void bvhtree_init(struct bvhtree *bvh)
{
	bvhtree_init_stat(bvh);
	octant_init(bvh, &bvh->root);
	bvh->leaves = NULL;
	bvh->octant_caches = NULL;
	bvh->num_octants = 0;
	bvh->num_caches = 0;
}

void bvhtree_deinit(struct bvhtree *bvh)
{
	int i;

	buf_destroy(bvh->leaves);
	/*
	 * No need to traverse the octants tree in order to destroy
	 * each octant, just destroy the cache.
	 */
	for (i = 0; i < bvh->num_caches; i++)
		buf_destroy(bvh->octant_caches[i].octants);
	free(bvh->octant_caches);
}

static int bvhtree_unmap(struct bvhtree *bvh)
{
	int i, ret = 0;

	if (bvh->leaves)
		ret = buf_unmap(bvh->leaves);
	for (i = 0; !ret && i < bvh->num_caches; i++) {
		if (bvh->octant_caches[i].octants)
			ret = buf_unmap(bvh->octant_caches[i].octants);
	}

	return ret;
}

static uint8_t init_child_extent(struct extent *child_extent,
				 const struct extent *parent_extent,
				 const struct extent_leaf *leaf)
{
	vec3_t leaf_centroid = extent_centroid(&leaf->e);
	vec3_t parent_centroid = extent_centroid(parent_extent);
	uint8_t child_index = 0;

	extent_init(child_extent);

	if (leaf_centroid.x > parent_centroid.x) {
		/* X right */
		child_index |= 1<<0;
		child_extent->d[0].near = parent_centroid.x;
		child_extent->d[0].far  = parent_extent->d[0].far;
	} else {
		/* X left */
		child_extent->d[0].near = parent_extent->d[0].near;
		child_extent->d[0].far  = parent_centroid.x;
	}
	if (leaf_centroid.y > parent_centroid.y) {
		/* Y top */
		child_index |= 1<<1;
		child_extent->d[1].near = parent_centroid.y;
		child_extent->d[1].far  = parent_extent->d[1].far;
	} else {
		/* Y bottom */
		child_extent->d[1].near = parent_extent->d[1].near;
		child_extent->d[1].far  = parent_centroid.y;
	}
	if (leaf_centroid.z > parent_centroid.z) {
		/* Z ahead */
		child_index |= 1<<2;
		child_extent->d[2].near = parent_centroid.z;
		child_extent->d[2].far  = parent_extent->d[2].far;
	} else {
		/* Z behind */
		child_extent->d[2].near = parent_extent->d[2].near;
		child_extent->d[2].far  = parent_centroid.z;
	}

	return child_index;
}

static struct octant *octant_cache_alloc(struct bvhtree *bvh,
					 struct scene *scene)
{
	int i_cache, i_octant;
	uint32_t size;

	i_cache = bvh->num_octants / OCTANT_CACHE_SIZE;
	i_octant = bvh->num_octants % OCTANT_CACHE_SIZE;

	if (bvh->num_caches <= i_cache) {
		struct octant_cache *caches;

		size = round_up(i_cache + 1, 16);
		caches = reallocarray(bvh->octant_caches, size, sizeof(*caches));
		if (!caches)
			return NULL;

		/* Zero out the rest, not yet occupied */
		memset(caches + bvh->num_caches, 0,
		       sizeof(*caches) * (size - bvh->num_caches));

		bvh->octant_caches = caches;
		bvh->num_caches = size;
	}
	if (!bvh->octant_caches[i_cache].octants) {
		__global struct octant *octants;

		size = sizeof(*octants) * OCTANT_CACHE_SIZE;
		octants = buf_allocate(scene->opencl, size);
		if (!octants)
			return NULL;

		bvh->octant_caches[i_cache].octants = octants;
	}

	return &bvh->octant_caches[i_cache].octants[i_octant];
}

static bool octant_can_split(const struct octant *octant, int depth)
{
	return octant_num_leaves(octant) >= OCTANT_MAX_LEAVES &&
		depth < OCTANT_MAX_DEPTH;
}

static struct octant *get_or_create_child_octant(struct bvhtree *bvh,
						 struct scene *scene,
						 struct octant *octant,
						 uint8_t i)
{
	struct octant *child_octant;

	if (!(child_octant = octant->octants[i])) {
		child_octant = octant_cache_alloc(bvh, scene);
		if (!child_octant)
			return NULL;

		octant_init(bvh, child_octant);
		octant->octants[i] = child_octant;
		bvh->num_octants++;
	}

	return child_octant;
}

static void octant_add_leaf(struct bvhtree *bvh, struct octant *octant,
			    struct extent_leaf *leaf, int depth)
{
	/* Add leaf extent to the leaf octant */
	assert(octant_is_leaf(octant));
	list_add_tail(&leaf->entry, &octant->leaves);
	octant_inc_leaves(octant);
	bvhtree_update_stat(bvh, octant, depth);
}

static int __octree_insert(struct bvhtree *bvh, struct scene *scene,
			   struct octant **poctant,
			   struct extent *octant_extent,
			   struct extent_leaf *leaf, int depth)
{
	struct octant *octant = *poctant;

	struct octant *child_octant;
	struct extent child_extent;
	uint8_t i_child;

	/* Expand octant extent with a leaf extent */
	extent_expand(&octant->extent, &leaf->e);

	if (octant_is_leaf(octant)) {
		struct extent child_extent_to_split;
		struct octant *child_to_split;

		LIST_HEAD(leaves);

		if (!octant_can_split(octant, depth)) {
			octant_add_leaf(bvh, octant, leaf, depth);
			*poctant = NULL;
			return 0;
		}

		/*
		 * Octant must be split, thus take all the leaves and
		 * reinsert them to children octants.
		 */
		list_splice_init(&octant->leaves, &leaves);
		list_add_tail(&leaf->entry, &leaves);
		octant_set_leaf(octant, false);
		child_to_split = NULL;
		while (!list_empty(&leaves)) {
			leaf = list_first_entry(&leaves, typeof(*leaf), entry);
			list_del(&leaf->entry);

			i_child = init_child_extent(&child_extent, octant_extent, leaf);
			child_octant = get_or_create_child_octant(bvh, scene, octant, i_child);
			if (!child_octant)
				return -ENOMEM;

			if (!octant_can_split(child_octant, depth + 1)) {
				extent_expand(&child_octant->extent, &leaf->e);
				octant_add_leaf(bvh, child_octant, leaf, depth + 1);
			} else {
				/* Child octant should be split */
				assert(!child_to_split);
				child_to_split = child_octant;
				child_extent_to_split = child_extent;
			}
		}
		*poctant = child_to_split;
		if (child_to_split)
			*octant_extent = child_extent_to_split;

		return 0;
	}

	i_child = init_child_extent(&child_extent, octant_extent, leaf);
	child_octant = get_or_create_child_octant(bvh, scene, octant, i_child);
	if (!child_octant)
		return -ENOMEM;

	/* Continue descent to this child octant on the next iteration */
	*poctant = child_octant;
	*octant_extent = child_extent;

	return 0;
}

static int
octree_insert(struct bvhtree *bvh, struct scene *scene,
	      const struct extent *root_extent,
	      struct extent_leaf *leaf)
{
	struct extent octant_extent = *root_extent;
	struct octant *octant;
	int ret = 0, depth = 0;

	for (octant = &bvh->root; !ret && octant; depth++)
		ret = __octree_insert(bvh, scene, &octant, &octant_extent,
				      leaf, depth);

	return ret;
}

static int
octree_build(struct bvhtree *bvh, struct scene *scene,
	     const struct extent *root_extent,
	     struct extent_leaf *leaves, uint32_t leaves_sz)
{
	struct bvhtree_stat *stat = &bvh->stat;
	int i, n, ret;

	for (i = 0; i < leaves_sz; i++) {
		ret = octree_insert(bvh, scene, root_extent, &leaves[i]);
		if (ret)
			return ret;
	}

	/* Iterate over all leaves in order to update statistics */
	while (!list_empty(&stat->octants_leaves)) {
		struct octant *octant;
		uint32_t num_leaves;

		octant = list_first_entry(&stat->octants_leaves, typeof(*octant), entry);
		list_del(&octant->entry);
		memset(octant->octants, 0, sizeof(octant->octants));

		num_leaves = octant_num_leaves(octant);
		for (i = 0, n = 10; i <= UPPER_BOUND; i++, n *= 10) {
			if (i < UPPER_BOUND && num_leaves >= n)
				continue;

			stat->num_octants[i] += num_leaves;
			break;
		}
	}
#if 0
	bvhtree_print_stat(bvh);
#endif

	bvh->leaves = leaves;
	bvh->alloc = &scene->alloc;

	return 0;
}

int bvhtree_build(struct bvhtree *bvh, struct scene *scene)
{
	struct extent_leaf *leaves;
	struct object *obj;
	struct extent scene_ext;
	uint32_t nr_triangles;
	int ret, tri;

	if (!scene->num_verts)
		/* Nothing to do */
		return 0;
	if (scene->num_verts % 3)
		/* Was not properly triangulated? */
		return -EINVAL;

	nr_triangles = scene->num_verts / 3;
	leaves = buf_allocate(scene->opencl, nr_triangles * sizeof(*leaves));
	if (!leaves)
		return -ENOMEM;

	extent_init(&scene_ext);

	/* Loop over meshes and create leaf extents */
	tri = 0;
	list_for_each_entry(obj, &scene->mesh_objects, entry) {
		struct triangle_mesh *mesh;
		int i;

		mesh = container_of(obj, typeof(*mesh), obj);

		/* Loop over triangles in mesh */
		for (i = 0; i < mesh->num_verts; i += 3, tri++) {
			struct extent_leaf *ext = &leaves[tri];

			extent_leaf_init(ext, mesh, i);
			/* Expand the whole scene */
			extent_expand(&scene_ext, &ext->e);
		}
	}
	ret = octree_build(bvh, scene, &scene_ext, leaves, nr_triangles);
	if (ret) {
		buf_destroy(leaves);
		return ret;
	}
	bvh->leaves = leaves;

	/* BVH ready, unmap before render */
	return bvhtree_unmap(bvh);
}
