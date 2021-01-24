#ifndef RAY_TRACE_H
#define RAY_TRACE_H

#include "scene.h"

/**
 * https://en.wikipedia.org/wiki/Halton_sequence
 */
static inline float halton_seq(int i, int b)
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
static inline bool solve_quadratic(float a, float b, float c, float *x0, float *x1)
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


static inline bool sphere_intersect(__global struct object *obj, const vec3_t *orig,
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

static inline void sphere_get_surface_props(__global struct object *obj,
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


static inline bool plane_intersect(__global struct object *obj, const vec3_t *orig,
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

static inline void plane_get_surface_props(__global struct object *obj,
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
 * Möller-Trumbore triangle intersection
 */
static inline bool
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

static inline bool __triangle_mesh_intersect(__global const struct triangle_mesh *mesh,
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

static inline bool triangle_mesh_intersect(__global struct object *obj,
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

static inline void triangle_mesh_get_surface_props(__global struct object *obj,
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


static inline void distant_light_illuminate(__global struct light *light, const vec3_t *orig,
					    vec3_t *dir, vec3_t *intensity, float *distance)
{
	__global struct distant_light *dlight;

	dlight = container_of(light, typeof(*dlight), light);

	*dir = dlight->dir;
	*intensity = v3_muls(dlight->light.color, dlight->light.intensity);
	*distance = INFINITY;
}


static inline void point_light_illuminate(__global struct light *light, const vec3_t *orig,
					  vec3_t *dir, vec3_t *intensity, float *distance)
{
	__global struct point_light *plight;
	float r_pow2;

	plight = container_of(light, typeof(*plight), light);

	*dir = v3_sub(*orig, plight->pos);
	r_pow2 = v3_dot(*dir, *dir);
	*distance = sqrt(r_pow2);
	dir->x /= *distance;
	dir->y /= *distance;
	dir->z /= *distance;
	*intensity = v3_muls(plight->light.color, plight->light.intensity);
	/* TODO: div by 0 */
	*intensity = v3_divs(*intensity, 4 * M_PI * r_pow2);
}

static inline bool
object_intersect(__global struct object *obj, const vec3_t *orig,
		 const vec3_t *dir, float *near, uint32_t *index,
		 vec2_t *uv)
{
#ifndef __OPENCL__
	return obj->ops.intersect(obj, orig, dir, near, index, uv);
#else
	/* OpenCL does not support function pointers, se la vie	 */
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

static inline void
object_get_surface_props(__global struct object *obj, const vec3_t *hit_point,
			 const vec3_t *dir, uint32_t index, const vec2_t *uv,
			 vec3_t *hit_normal, vec2_t *hit_tex_coords)
{
#ifndef __OPENCL__
	obj->ops.get_surface_props(obj, hit_point, dir, index,
				   uv, hit_normal, hit_tex_coords);
#else
	/* OpenCL does not support function pointers, se la vie	 */
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

static inline float object_pattern(__global struct object *obj,
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

static inline void
light_illuminate(__global struct light *light, const vec3_t *orig,
		 vec3_t *dir, vec3_t *intensity, float *distance)
{
#ifndef __OPENCL__
	light->ops.illuminate(light, orig, dir, intensity, distance);
#else
	/* OpenCL does not support function pointers, se la vie	 */
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

static inline bool bvhtree_intersect(__global const struct bvhtree *bvh,
				     const vec3_t *orig, const vec3_t *dir,
				     struct intersection *isect,
				     enum ray_type ray_type)
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

	ret = octant_queue_init(&queue, bvh->alloc);
	if (ret) {
		assert(0);
		return false;
	}

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
	ret = octant_queue_deinit(&queue);
	assert(!ret);

	return !!isect->hit_object;
}

static inline void
ray_intersect_objects(__global struct scene *scene, struct list_head *objects,
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

static inline bool
ray_trace(__global struct scene *scene, const vec3_t *orig, const vec3_t *dir,
	  struct intersection *isect, enum ray_type ray_type)
{
	isect->hit_object = NULL;
	isect->near = INFINITY;

	/* Trace meshes */
	if (scene->dont_use_bvh)
		ray_intersect_objects(scene, &scene->mesh_objects, orig, dir,
				      isect, ray_type);
	else
		bvhtree_intersect(&scene->bvhtree, orig, dir, isect, ray_type);

	/* Trace other objects */
	ray_intersect_objects(scene, &scene->notmesh_objects, orig, dir,
			      isect, ray_type);

	return !!isect->hit_object;
}

/**
 * Compute reflection direction
 */
static inline vec3_t reflect(const vec3_t *I, const vec3_t *N)
{
	float dot = v3_dot(*I, *N);

	return v3_sub(*I, v3_muls(*N, 2 * dot));
}

/**
 * Compute refraction direction
 */
static inline vec3_t refract(const vec3_t *I, const vec3_t *N, float ior)
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
 * Evaluate Fresnel equation (ration of reflected light for a
 * given incident direction and surface normal)
 */
static inline float fresnel(const vec3_t *I, const vec3_t *N, float ior)
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


static inline bool __ray_cast(struct ray_cast_input *in, struct ray_cast_output *out,
			      struct ray_cast_state *s)
{
	struct intersection isect;

	vec3_t hit_point, hit_normal, hit_color, dir = in->dir;
	vec2_t hit_tex_coords;
	bool hit;

	/* Continue execution if was yielded */
	switch (s->type) {
	case RAY_CAST_REFLECT_YIELD:
		goto reflect_continue;
	case RAY_CAST_RR_REFRACT_YIELD:
		goto rr_refract_continue;
	case RAY_CAST_RR_REFLECT_YIELD:
		goto rr_reflect_continue;
	default:
		break;
	}

	/* Update stat */
	atomic64_inc(&in->scene->stat.rays);

	hit = ray_trace(in->scene, &in->orig, &dir, &isect, PRIMARY_RAY);
	if (!hit) {
		out->color = in->scene->backcolor;
		return false;
	}

	hit_color = vec3(0.0f, 0.0f, 0.0f);

	/* Evaluate surface properties (P, N, texture coordinates, etc.) */
	hit_point = v3_add(in->orig, v3_muls(dir, isect.near));
	object_get_surface_props(isect.hit_object, &hit_point, &dir, isect.index,
				 &isect.uv, &hit_normal, &hit_tex_coords);
	switch (isect.hit_object->material) {
	case MATERIAL_PHONG: {
		/*
		 * Light loop (loop over all lights in the scene
		 * and accumulate their contribution)
		 */
		vec3_t diffuse, specular;
		__global struct light *light;

		diffuse = specular = vec3(0.0f, 0.0f, 0.0f);

		list_for_each_entry(light, &in->scene->lights, entry) {
			vec3_t light_dir, light_intensity;
			vec3_t point, rev_light_dir, R;
			vec3_t rev_dir, diff, spec;

			struct intersection isect_shadow;
			float near, pattern, p;
			bool obstacle;

			light_illuminate(light, &hit_point, &light_dir,
					 &light_intensity, &near);

			point = v3_add(hit_point, v3_muls(hit_normal, in->scene->bias));
			rev_light_dir = v3_muls(light_dir, -1.0f);

			obstacle = !!ray_trace(in->scene, &point, &rev_light_dir,
					       &isect_shadow, SHADOW_RAY);
			if (obstacle)
				/* Light is not visible, object is hit, thus shadow */
				continue;

			/* compute the diffuse component */
			pattern = object_pattern(isect.hit_object, &hit_tex_coords);
			diff = v3_muls(light_intensity,
				       pattern * isect.hit_object->albedo *
				       MAX(0.0f, v3_dot(hit_normal, rev_light_dir)));
			diffuse = v3_add(diffuse, diff);

			/*
			 * compute the specular component
			 * what would be the ideal reflection direction for this
			 * light ray
			 */
			R = reflect(&light_dir, &hit_normal);

			rev_dir = v3_muls(dir, -1.0f);

			p = powf(MAX(0.0f, v3_dot(R, rev_dir)), isect.hit_object->n);
			spec = v3_muls(light_intensity, p);
			specular = v3_add(specular, spec);
		}
		/* Compute the whole light contribution */
		diffuse = v3_mul(diffuse, isect.hit_object->Kd);
		specular = v3_mul(specular, isect.hit_object->Ks);
		hit_color = v3_add(diffuse, specular);
		if (isect.hit_object->r)
			/* Object is reflective */
			goto calculate_reflect;
		break;
	}
	case MATERIAL_REFLECT: {
		vec3_t reflect_dir;
		vec3_t color;
calculate_reflect:
		reflect_dir = reflect(&dir, &hit_normal);

		hit_point = v3_add(hit_point, v3_muls(hit_normal, in->scene->bias));

		in->orig = hit_point;
		in->dir = reflect_dir;
		s->reflect.hit_color = hit_color;
		s->reflect.hit_object = isect.hit_object;
		s->type = RAY_CAST_REFLECT_YIELD;
		return true;
		/* color = ray_cast(&hit_point, &reflect_dir); */
reflect_continue:
		color = v3_muls(out->color, s->reflect.hit_object->r);
		hit_color = v3_add(s->reflect.hit_color, color);
		break;
	}
	case MATERIAL_REFLECT_REFRACT: {
		vec3_t refract_color = vec3(0.0f, 0.0f, 0.0f);
		vec3_t reflect_color = vec3(0.0f, 0.0f, 0.0f);
		vec3_t reflect_orig, reflect_dir, bias;
		bool outside;
		float kr;

		kr = fresnel(&dir, &hit_normal, isect.hit_object->ior);
		outside = v3_dot(dir, hit_normal) < 0.0f;
		bias = v3_muls(hit_normal, in->scene->bias);

		/* compute refraction if it is not a case of total internal reflection */
		if (kr < 1.0f) {
			vec3_t refract_orig, refract_dir;

			refract_dir = refract(&dir, &hit_normal, isect.hit_object->ior);
			refract_dir = v3_norm(refract_dir);

			refract_orig = outside ?
				v3_sub(hit_point, bias) :
				v3_add(hit_point, bias);

			in->orig = refract_orig;
			in->dir = refract_dir;
			s->rr_refract.kr = kr;
			s->rr_refract.hit_normal = hit_normal;
			s->rr_refract.outside = outside;
			s->rr_refract.bias = bias;
			s->rr_refract.hit_point = hit_point;
			s->rr_refract.dir = dir;
			s->type = RAY_CAST_RR_REFRACT_YIELD;
			return true;
			/* refract_color = ray_cast(&refract_orig, &refract_dir); */
rr_refract_continue:
			kr = s->rr_refract.kr;
			hit_normal = s->rr_refract.hit_normal;
			outside = s->rr_refract.outside;
			bias = s->rr_refract.bias;
			hit_point = s->rr_refract.hit_point;
			dir = s->rr_refract.dir;

			refract_color = v3_muls(out->color, 1 - kr);
		}
		reflect_dir = reflect(&dir, &hit_normal);
		reflect_dir = v3_norm(reflect_dir);

		reflect_orig = outside ?
			v3_add(hit_point, bias) :
			v3_sub(hit_point, bias);

		in->orig = reflect_orig;
		in->dir = reflect_dir;
		s->rr_reflect.refract_color = refract_color;
		s->rr_reflect.kr = kr;
		s->type = RAY_CAST_RR_REFLECT_YIELD;
		return true;
		/* reflect_color = ray_cast(&reflect_orig, &reflect_dir); */
rr_reflect_continue:
		reflect_color = v3_muls(out->color, s->rr_reflect.kr);

		hit_color = v3_add(reflect_color, s->rr_reflect.refract_color);
		break;
	}
	default:
		hit_color = in->scene->backcolor;
		break;
	}

	out->color = hit_color;
	return false;
}

static inline vec3_t ray_cast(__global struct scene *scene,
			      __global struct ray_cast_state *ray_states,
			      const vec3_t *orig, const vec3_t *dir)
{
	__global struct ray_cast_state *s = ray_states;
	struct ray_cast_input in = {
		.scene = scene,
		.orig = *orig,
		.dir = *dir
	};
	struct ray_cast_output out = {
		.color = vec3(0.0f, 0.0f, 0.0f)
	};
	int depth;

	/*
	 * Flatten recursion with a simple loop. Since we can cast rays on
	 * OpenCL we can't rely on a big stack support on GPU.
	 */
	s->type = RAY_CAST_CALL;
	depth = 0;
	while (1) {
		bool yielded = __ray_cast(&in, &out, s);
		if (yielded) {
			if (depth + 1 < scene->ray_depth) {
				/* Take next state and prepare for call */
				s = &ray_states[++depth];

				/*
				 * Prepare for next ray cast, input is already set
				 * by the previous ray cast.
				 */
				s->type = RAY_CAST_CALL;
				continue;
			}
			/* Maximum depth is reached */
			out.color = scene->backcolor;

			/* Pretend call is completed and fall through */
		}
		if (!depth)
			/* Top is reached */
			return out.color;

		/*
		 * Take previous state, output is already set
		 * by the previous ray cast
		 */
		s = &ray_states[--depth];
	}

	/* Unreachable line */
	return scene->backcolor;
}

static inline vec3_t ray_cast_for_pixel(__global struct scene *scene,
					const vec3_t *orig, int ix, int iy,
					float scale, float img_ratio)
{
	vec3_t color, dir;
	float x, y;
	int n;

	color = vec3(0.0f, 0.0f, 0.0f);
	for (n = 1; n <= scene->samples_per_pixel; n++) {
		__global struct ray_cast_state *ray_states;
		uint32_t ray_states_off;

		/* Repeatable jitter */
		x = ix + halton_seq(n, 3);
		y = iy + halton_seq(n, 2);

		x = (2.0f * x / scene->width - 1.0f) * img_ratio * scale;
		y = (1.0f - 2.0f * y / scene->height) * scale;

		dir = m4_mul_dir(scene->c2w, vec3(x, y, -1.0f));
		dir = v3_norm(dir);

		ray_states_off = (iy * scene->width + ix) * scene->ray_depth;
		ray_states = scene->ray_states + ray_states_off;
		color = v3_add(color, ray_cast(scene, ray_states, orig, &dir));
	}
	color = v3_divs(color, scene->samples_per_pixel);

	return color;
}

static inline void color_vec_to_rgba32(const vec3_t *color, struct rgba *rgb)
{
	*rgb = (struct rgba) {
		.r = (255 * clamp(0.0f, 1.0f, color->x)),
		.g = (255 * clamp(0.0f, 1.0f, color->y)),
		.b = (255 * clamp(0.0f, 1.0f, color->z))
	};
}

#endif /* RAY_TRACE_H */
