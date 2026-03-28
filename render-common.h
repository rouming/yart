#ifndef RENDER_COMMON_H
#define RENDER_COMMON_H

#include "scene.h"

/**
 * https://en.wikipedia.org/wiki/Halton_sequence
 */
__accelerated static inline float halton_seq(int i, int b)
{
	float r = 0.0f;
	float v = 1.0f;
	float binv = 1.0f / b;

	while (i > 0) {
		v *= binv;
		r += v * (i % b);
		i /= b;
	}
	return r;
}

/**
 * Compute the roots of a quadratic equation
 */
__accelerated static inline bool solve_quadratic(float a, float b, float c, float *x0, float *x1)
{
	float discr = b * b - 4 * a * c;
	if (discr < 0)
		return false;

	if (discr == 0) {
		*x0 = *x1 = - 0.5 * b / a;
	} else {
		float q = (b > 0) ?
			-0.5f * (b + sqrtf(discr)) :
			-0.5f * (b - sqrtf(discr));
		*x0 = q / a;
		*x1 = c / q;
	}

	return true;
}


__accelerated static inline bool sphere_intersect(__global struct object *obj, const vec3_t *orig,
				    const vec3_t *dir, float *near, uint32_t *index,
				    vec2_t *uv)
{
	__global struct sphere *sphere;

	sphere = container_of(obj, typeof(*sphere), obj);

	/* not used in sphere */
	*index = 0;
	*uv = vec2(0.0f, 0.0f);

	/* solutions for t if the ray intersects */
	float t0, t1;

	/* analytic solution */
	vec3_t L = v3_sub(*orig, sphere->center);
	float a = v3_dot(*dir, *dir);
	float b = 2 * v3_dot(*dir, L);
	float c = v3_dot(L, L) - sphere->radius_pow2;

	if (!solve_quadratic(a, b, c, &t0, &t1))
		return false;
	if (t0 > t1) {
		SWAP(t0, t1);
	}
	if (t0 < 0) {
		/* if t0 is negative, let's use t1 instead */
		t0 = t1;
		if (t0 < 0)
			/* both t0 and t1 are negative */
			return false;
	}
	*near = t0;

	return true;
}

__accelerated static inline void sphere_get_surface_props(__global struct object *obj,
					    const vec3_t *hit_point,
					    const vec3_t *dir, uint32_t index,
					    const vec2_t *uv, vec3_t *hit_normal,
					    vec2_t *hit_tex_coords)
{
	__global struct sphere *sphere;

	sphere = container_of(obj, typeof(*sphere), obj);

	*hit_normal = v3_norm(v3_sub(*hit_point, sphere->center));

	/*
	 * In this particular case, the normal is similar to a point on a unit sphere
	 * centred around the origin. We can thus use the normal coordinates to compute
	 * the spherical coordinates of Phit.
	 * atan2 returns a value in the range [-pi, pi] and we need to remap it to range [0, 1]
	 * acosf returns a value in the range [0, pi] and we also need to remap it to the range [0, 1]
	 */
	hit_tex_coords->x = (1.0f + atan2f(hit_normal->z, hit_normal->x) / M_PI) * 0.5f;
	hit_tex_coords->y = acosf(hit_normal->y) / M_PI;
}


__accelerated static inline bool plane_intersect(__global struct object *obj, const vec3_t *orig,
				   const vec3_t *dir, float *near, uint32_t *index,
				   vec2_t *uv)
{
	__global struct plane *plane;

	/* not used in plane */
	*index = 0;
	*uv = vec2(0.0f, 0.0f);

	plane = container_of(obj, typeof(*plane), obj);

	/*
	 * Plane in a general form: Ax + By + Cz + d = 0,
	 * where Normal = (A, B, C), thus:
	 *   dot(Normal, orig + dist * dir) + d = 0
	 *   dot(Normal, orig) + dot(Normal, dist * dir) + d = 0
	 *   dot(Normal, orig) + dist * dot(Normal, dir) + d = 0
	 *   dist = -(dot(Normal, orig) + d) / dot(Normal, dir)
	 */
	*near = -(v3_dot(plane->normal, *orig) + plane->d) /
		  v3_dot(plane->normal, *dir);

	/* If negative plane is behind the camera */
	return *near > 0;
}

__accelerated static inline void plane_get_surface_props(__global struct object *obj,
					   const vec3_t *hit_point,
					   const vec3_t *dir, uint32_t index,
					   const vec2_t *uv, vec3_t *hit_normal,
					   vec2_t *hit_tex_coords)
{
	__global struct plane *plane;

	plane = container_of(obj, typeof(*plane), obj);

	/* Project 3D point to 2D plane */
	hit_tex_coords->x = v3_dot(plane->b1, *hit_point);
	hit_tex_coords->y = v3_dot(plane->b2, *hit_point);

	*hit_normal = plane->normal;
}


/**
 * Moller-Trumbore triangle intersection
 */
__accelerated static inline bool
triangle_intersect(const vec3_t *orig, const vec3_t *dir,
		   __global const vec3_t *v0,
		   __global const vec3_t *v1,
		   __global const vec3_t *v2,
		   float *t, float *u, float *v)
{
	vec3_t v0v1 = v3_sub(*v1, *v0);
	vec3_t v0v2 = v3_sub(*v2, *v0);
	vec3_t pvec = v3_cross(*dir, v0v2);
	float det = v3_dot(v0v1, pvec);
	vec3_t tvec, qvec;

	float inv_det;

	/* ray and triangle are parallel if det is close to 0 */
	if (fabs(det) < EPSILON)
		return false;

	inv_det = 1 / det;

	tvec = v3_sub(*orig, *v0);
	*u = v3_dot(tvec, pvec) * inv_det;
	if (*u < 0 || *u > 1)
		return false;

	qvec = v3_cross(tvec, v0v1);
	*v = v3_dot(*dir, qvec) * inv_det;
	if (*v < 0 || *u + *v > 1)
		return false;

	*t = v3_dot(v0v2, qvec) * inv_det;

	return (*t > 0);
}

__accelerated static inline bool __triangle_mesh_intersect(__global const struct triangle_mesh *mesh,
					     const vec3_t *orig, const vec3_t *dir,
					     float *near, uint32_t i, vec2_t *uv)
{
	__global const vec3_t *vertices = mesh->vertices;
	__global const vec3_t *v0 = &vertices[i + 0];
	__global const vec3_t *v1 = &vertices[i + 1];
	__global const vec3_t *v2 = &vertices[i + 2];

	float t = INFINITY, u, v;

	if (triangle_intersect(orig, dir, v0, v1, v2, &t, &u, &v) &&
	    t < *near) {
		*near = t;
		uv->x = u;
		uv->y = v;
		return true;
	}

	return false;
}

__accelerated static inline bool triangle_mesh_intersect(__global struct object *obj,
					   const vec3_t *orig, const vec3_t *dir,
					   float *near, uint32_t *index, vec2_t *uv)
{
	__global struct triangle_mesh *mesh;

	uint32_t i;
	bool isect;

	mesh = container_of(obj, typeof(*mesh), obj);

	isect = false;
	for (i = 0; i < mesh->num_verts; i += 3) {
		if (__triangle_mesh_intersect(mesh, orig, dir, near, i, uv)) {
			*index = i / 3;
			isect = true;
		}
	}

	return isect;
}

__accelerated static inline void triangle_mesh_get_surface_props(__global struct object *obj,
						   const vec3_t *hit_point,
						   const vec3_t *dir, uint32_t index,
						   const vec2_t *uv, vec3_t *hit_normal,
						   vec2_t *hit_tex_coords)
{
	__global struct triangle_mesh *mesh;
	__global const vec2_t *sts;
	vec2_t st0, st1, st2;

	mesh = container_of(obj, typeof(*mesh), obj);
	if (mesh->smooth_shading) {
		/* vertex normal */
		__global const vec3_t *normals = mesh->normals;
		vec3_t n0 = normals[index * 3 + 0];
		vec3_t n1 = normals[index * 3 + 1];
		vec3_t n2 = normals[index * 3 + 2];

		n0 = v3_muls(n0, 1 - uv->x - uv->y);
		n1 = v3_muls(n1, uv->x);
		n2 = v3_muls(n2, uv->y);

		*hit_normal = v3_add(n2, v3_add(n0, n1));
	} else {
		/* face normal */
		__global const vec3_t *vertices = mesh->vertices;
		vec3_t v0 = vertices[index * 3 + 0];
		vec3_t v1 = vertices[index * 3 + 1];
		vec3_t v2 = vertices[index * 3 + 2];

		vec3_t v1v0 = v3_sub(v1, v0);
		vec3_t v2v0 = v3_sub(v2, v0);

		*hit_normal = v3_cross(v1v0, v2v0);
	}

	/*
	 * doesn't need to be normalized as the N's are
	 * normalized but just for safety
	 */
	*hit_normal = v3_norm(*hit_normal);

	/* texture coordinates */
	sts = mesh->sts;
	st0 = sts[index * 3 + 0];
	st1 = sts[index * 3 + 1];
	st2 = sts[index * 3 + 2];

	st0 = v2_muls(st0, 1 - uv->x - uv->y);
	st1 = v2_muls(st1, uv->x);
	st2 = v2_muls(st2, uv->y);

	*hit_tex_coords = v2_add(st2, v2_add(st0, st1));
}


__accelerated static inline void distant_light_illuminate(__global struct light *light, const vec3_t *orig,
					    vec3_t *dir, vec3_t *intensity, float *distance)
{
	__global struct distant_light *dlight;

	dlight = container_of(light, typeof(*dlight), light);

	*dir = dlight->dir;
	*intensity = v3_muls(dlight->light.color, dlight->light.intensity);
	*distance = INFINITY;
}


__accelerated static inline void point_light_illuminate(__global struct light *light, const vec3_t *orig,
					  vec3_t *dir, vec3_t *intensity, float *distance)
{
	__global struct point_light *plight;
	float r_pow2;

	plight = container_of(light, typeof(*plight), light);

	*dir = v3_sub(*orig, plight->pos);
	r_pow2 = v3_dot(*dir, *dir);
	if (r_pow2 < EPSILON)
		r_pow2 = EPSILON;
	*distance = sqrt(r_pow2);
	dir->x /= *distance;
	dir->y /= *distance;
	dir->z /= *distance;
	*intensity = v3_muls(plight->light.color, plight->light.intensity);
	*intensity = v3_divs(*intensity, 4 * M_PI * r_pow2);
}

__accelerated static inline bool
object_intersect(__global struct object *obj, const vec3_t *orig,
		 const vec3_t *dir, float *near, uint32_t *index,
		 vec2_t *uv)
{
#if !defined(__OPENCL__) && !defined(__CUDA_ARCH__)
	return obj->ops->intersect(obj, orig, dir, near, index, uv);
#else
	/* OpenCL/CUDA device: no function pointers */
	switch (obj->type) {
	case SPHERE_OBJECT:
		return sphere_intersect(obj, orig, dir, near, index, uv);
	case PLANE_OBJECT:
		return plane_intersect(obj, orig, dir, near, index, uv);
	case MESH_OBJECT:
		return triangle_mesh_intersect(obj, orig, dir, near, index, uv);
	default:
		/* Hm .. */
		printf("%s: unknown object %d\n", __func__, obj->type);
		return false;
	}
#endif
}

__accelerated static inline void
object_get_surface_props(__global struct object *obj, const vec3_t *hit_point,
			 const vec3_t *dir, uint32_t index, const vec2_t *uv,
			 vec3_t *hit_normal, vec2_t *hit_tex_coords)
{
#if !defined(__OPENCL__) && !defined(__CUDA_ARCH__)
	obj->ops->get_surface_props(obj, hit_point, dir, index,
				    uv, hit_normal, hit_tex_coords);
#else
	/* OpenCL/CUDA device: no function pointers */
	switch (obj->type) {
	case SPHERE_OBJECT:
		sphere_get_surface_props(obj, hit_point, dir, index,
					 uv, hit_normal, hit_tex_coords);
		return;
	case PLANE_OBJECT:
		plane_get_surface_props(obj, hit_point, dir, index,
					uv, hit_normal, hit_tex_coords);
		return;
	case MESH_OBJECT:
		triangle_mesh_get_surface_props(obj, hit_point, dir, index,
						uv, hit_normal, hit_tex_coords);
		return;
	default:
		/* Hm .. */
		printf("%s: unknown object %d\n", __func__, obj->type);
		return;
	}
#endif
}

__accelerated static inline float object_pattern(__global struct object *obj,
				   vec2_t *hit_tex_coords)
{
	float angle, co, si, s, t, scale;

	if (obj->pattern.type == PATTERN_UNKNOWN)
		return 1.0f;

	angle = deg2rad(obj->pattern.angle);
	co = cos(angle);
	si = sin(angle);
	s = hit_tex_coords->x * co - hit_tex_coords->y * si;
	t = hit_tex_coords->y * co + hit_tex_coords->x * si;
	scale = 1.0f / obj->pattern.scale;

	if (obj->pattern.type == PATTERN_CHECK)
		return (modulo(s * scale) < 0.5) ^ (modulo(t * scale) < 0.5);
	else if (obj->pattern.type == PATTERN_LINE)
		return (modulo(s * scale) < 0.5);

	/* Hm, unreachable line actually */
	return 1.0f;
}

__accelerated static inline void
light_illuminate(__global struct light *light, const vec3_t *orig,
		 vec3_t *dir, vec3_t *intensity, float *distance)
{
#if !defined(__OPENCL__) && !defined(__CUDA_ARCH__)
	light->ops->illuminate(light, orig, dir, intensity, distance);
#else
	/* OpenCL/CUDA device: no function pointers */
	switch (light->type) {
	case DISTANT_LIGHT:
		distant_light_illuminate(light, orig, dir, intensity, distance);
		return;
	case POINT_LIGHT:
		point_light_illuminate(light, orig, dir, intensity, distance);
		return;
	default:
		/* Hm .. */
		printf("%s: unknown light %d\n", __func__, light->type);
		return;
	}
#endif
}

__accelerated static inline bool bvhtree_intersect(__global const struct bvhtree *bvh,
				     const vec3_t *orig, const vec3_t *dir,
				     struct intersection *isect,
				     enum ray_type ray_type,
				     __global struct octant_queue_entry *q_entries,
				     uint32_t q_depth)
{
	__global const struct octant *octant;

	struct octant_queue queue;

	float numerators[NR_PLANE_NORMALS];
	float denominators[NR_PLANE_NORMALS];
	float t_near, t_far, t_hit, t_octant;
	int i, ret;
	vec2_t uv;

	/* Precompute dot products for plane-set normals, see equation above */
	for (i = 0; i < NR_PLANE_NORMALS; i++) {
		numerators[i]   = v3_dot(*orig, plane_set_normals[i]);
		denominators[i] = v3_dot(*dir,  plane_set_normals[i]);
	}

	t_near = 0.0f;
	t_far  = INFINITY;

	if (!extent_intersect(&bvh->root.extent, numerators, denominators,
			      &t_near, &t_far)
	    /* XXX */
	    || t_far < 0.0f) {
		return false;
	}

	octant_queue_init(&queue, q_entries, q_depth);

	octant = &bvh->root;
	t_octant = 0.0f;
	t_hit = t_far;

	do {
		if (t_octant >= t_hit)
			break;

		if (octant_is_leaf(octant)) {
			__global struct extent_leaf *leaf;

			list_for_each_entry(leaf, &octant->leaves, entry) {
				if (ray_type == SHADOW_RAY &&
				    leaf->mesh->obj.material == MATERIAL_REFLECT_REFRACT)
					/* No shadow for objects which reflect-refract */
					continue;

				if (__triangle_mesh_intersect(leaf->mesh, orig, dir, &t_hit,
							      leaf->index, &uv)) {
					isect->hit_object = &leaf->mesh->obj;
					isect->near = t_hit;
					/* XXX: sometimes triangles, sometimes vertices */
					isect->index = leaf->index / 3;
					isect->uv = uv;
				}
			}
			continue;
		}

		/* Not a leaf octant, continue descent to children */
		for (i = 0; i < ARRAY_SIZE(octant->octants); i++) {
			float t_near_child, t_far_child;
			__global struct octant *child;

			if (!(child = octant->octants[i]))
				continue;

			t_near_child = 0.0f;
			t_far_child  = t_far;

			if (extent_intersect(&child->extent, numerators, denominators,
					     &t_near_child, &t_far_child)) {
				/* XXX */
				float t = (t_near_child < 0.0f && t_far_child >= 0.0f) ?
					t_far_child : t_near_child;

				ret = octant_queue_insert(&queue, child, t);
				if (ret) {
					isect->hit_object = NULL;
					assert(0);
					goto out;
				}
			}
		}
	} while ((octant = octant_queue_pop_first(&queue, &t_octant)));

out:
	octant_queue_deinit(&queue);

	return !!isect->hit_object;
}

__accelerated static inline void
ray_intersect_objects(__global struct scene *scene, __global struct list_head *objects,
		      const vec3_t *orig, const vec3_t *dir,
		      struct intersection *isect, enum ray_type ray_type)
{
	__global struct object *obj;

	/* Trace objects */
	list_for_each_entry(obj, objects, entry) {
		float near = INFINITY;
		uint32_t index = 0;
		vec2_t uv;

		if (ray_type == SHADOW_RAY &&
		    obj->material == MATERIAL_REFLECT_REFRACT)
			/* No shadow for objects which reflect-refract */
			continue;

		if (object_intersect(obj, orig, dir, &near, &index, &uv) &&
		    near < isect->near) {
			isect->hit_object = obj;
			isect->near = near;
			isect->index = index;
			isect->uv = uv;
		}
	}
}

__accelerated static inline bool
ray_trace(__global struct scene *scene, const vec3_t *orig, const vec3_t *dir,
	  struct intersection *isect, enum ray_type ray_type,
	  __global struct octant_queue_entry *q_entries, uint32_t q_depth)
{
	isect->hit_object = NULL;
	isect->near = INFINITY;

	/* Trace meshes */
	if (scene->dont_use_bvh)
		ray_intersect_objects(scene, &scene->mesh_objects, orig, dir,
				      isect, ray_type);
	else
		bvhtree_intersect(&scene->bvhtree, orig, dir, isect, ray_type,
				  q_entries, q_depth);

	/* Trace other objects */
	ray_intersect_objects(scene, &scene->notmesh_objects, orig, dir,
			      isect, ray_type);

	return !!isect->hit_object;
}

/**
 * Compute reflection direction
 */
__accelerated static inline vec3_t reflect(const vec3_t *I, const vec3_t *N)
{
	float dot = v3_dot(*I, *N);

	return v3_sub(*I, v3_muls(*N, 2 * dot));
}

/**
 * Compute refraction direction
 */
__accelerated static inline vec3_t refract(const vec3_t *I, const vec3_t *N, float ior)
{
	float cosi = clamp(-1.0f, 1.0f, v3_dot(*I, *N));
	float etai = 1, etat = ior, eta, k;
	vec3_t n = *N;

	if (cosi < 0) {
		cosi = -cosi;
	} else {
		SWAP(etai, etat);
		n = v3_muls(*N, -1.0f);
	}

	eta = etai / etat;
	k = 1 - eta * eta * (1 - cosi * cosi);

	if (k < 0) {
		return vec3(0.0f, 0.0f, 0.0f);
	} else {
		vec3_t Ieta = v3_muls(*I, eta);
		vec3_t Neta = v3_muls(n, eta * cosi - sqrtf(k));

		return v3_add(Ieta, Neta);
	}
}

/**
 * Evaluate Fresnel equation (ratio of reflected light for a
 * given incident direction and surface normal)
 */
__accelerated static inline float fresnel(const vec3_t *I, const vec3_t *N, float ior)
{
	float cosi = clamp(-1.0f, 1.0f, v3_dot(*I, *N));
	float etai = 1.0f, etat = ior;
	float sint, kr;

	if (cosi > 0)
		SWAP(etai, etat);

	/* Compute sini using Snell's law */
	sint = etai / etat * sqrtf(MAX(0.0f, 1.0f - cosi * cosi));

	/* Total internal reflection */
	if (sint >= 1.0f) {
		kr = 1.0f;
	} else {
		float cost, Rs, Rp;

		cost = sqrtf(MAX(0.0f, 1.0f - sint * sint));
		cosi = fabsf(cosi);
		Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}
	/*
	 * As a consequence of the conservation of energy,
	 * transmittance is given by:
	 * kt = 1 - kr;
	 */

	return kr;
}

__accelerated static inline void color_vec_to_rgba32(const vec3_t *color, __global struct rgba *rgb)
{
	rgb->r = (uint8_t)(255 * clamp(0.0f, 1.0f, color->x));
	rgb->g = (uint8_t)(255 * clamp(0.0f, 1.0f, color->y));
	rgb->b = (uint8_t)(255 * clamp(0.0f, 1.0f, color->z));
	rgb->a = 255;
}

#endif /* RENDER_COMMON_H */
