// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * YART (Yet Another Ray Tracing) boosted by OpenCL
 * Copyright (C) 2020,2021 Roman Penyaev
 *
 * Based on lessons from scratchapixel.com and pbr-book.org
 *
 * Roman Penyaev <r.peniaev@gmail.com>
 */

#ifndef __OPENCL__
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <errno.h>
#include <endian.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <SDL2/SDL.h>

#define __global

#else /* __OPENCL__ */

#define sinf  sin
#define cosf  cos
#define tanf  tan
#define acosf acos
#define fabsf fabs
#define sqrtf sqrt
#define powf  pow

struct scene;

typedef unsigned int uint32_t;
typedef int	     int32_t;

typedef unsigned char uint8_t;

#endif /* __OPENCL__ */

#ifndef offsetof
#define offsetof(t,m) __builtin_offsetof(t, m)
#endif

#ifndef container_of
#define container_of(ptr, type, member) ({			\
	const typeof( ((type*)0)->member )* __mptr = (ptr);	\
	(type*)( (uintptr_t)__mptr - offsetof(type, member));	\
})
#endif

#include "list.h"
#include "math_3d.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SWAP(a, b) do { typeof(a) temp = a; a = b; b = temp; } while (0)

#define EPSILON 1e-8

struct opencl;
struct sdl;

struct rgba {
	union {
		struct {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
			uint8_t a, b, g, r;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
			uint8_t r, g, b, a;
#else
#error "Unknown endianess"
#endif
		};
		uint32_t rgba8888;
	};
};

struct camera {
	vec3_t pos;
	vec3_t dir;
	float  pitch;
	float  yaw;
};

struct scene {
	uint32_t width;
	uint32_t height;
	float	 fov;
	vec3_t	 back_color;
	mat4_t	 c2w;
	float	 bias;
	uint32_t max_depth;
	struct camera cam;
	__global struct rgba *framebuffer;
	struct opencl *opencl;
	struct sdl    *sdl;

	struct list_head objects;
	struct list_head lights;
};

#ifndef __OPENCL__

struct opencl {
	cl_context	 context;
	cl_device_id	 device_id;
	cl_command_queue queue;
	cl_program	 program;
	cl_kernel	 kernel;
};

struct sdl {
	SDL_Window   *window;
	SDL_Renderer *renderer;
	SDL_Texture  *screen;
};

enum {
	BUF_MAP_WRITE = 1<<0,
	BUF_MAP_READ  = 1<<1,
	BUF_ZERO      = 1<<2,
};

struct buf_region {
	struct opencl *opencl;
	uint32_t      size;
};

static void *__buf_allocate(struct opencl *opencl, size_t sz, uint32_t flags)
{
	struct buf_region *reg;
	void *ptr;
	int ret;

	if (!sz)
		return NULL;

	if (opencl) {
		reg = clSVMAlloc(opencl->context,
				 CL_MEM_READ_WRITE /* | CL_MEM_SVM_FINE_GRAIN_BUFFER */,
				 sz + 16, 0);
		if (reg && (flags & (BUF_MAP_WRITE | BUF_MAP_READ))) {
			cl_map_flags cl_flags = 0;

			if (flags & BUF_MAP_WRITE)
				cl_flags |= CL_MAP_WRITE;
			if (flags & BUF_MAP_READ)
				cl_flags |= CL_MAP_READ;

			ret = clEnqueueSVMMap(opencl->queue, CL_TRUE, cl_flags,
					      (void *)reg + 16, sz, 0,
					      NULL, NULL);
			if (ret) {
				clSVMFree(opencl->context, reg);
				return NULL;
			}
		}
	} else {
		reg = malloc(sz + 16);
	}
	if (!reg)
		return NULL;

	ptr = (void *)reg + 16;

	if (flags & BUF_ZERO)
		memset(ptr, 0, sz);

	reg->opencl = opencl;
	reg->size = sz;

	return ptr;
}

static void *buf_allocate(struct opencl *opencl, size_t sz)
{
	return __buf_allocate(opencl, sz, BUF_ZERO | BUF_MAP_WRITE);
}

static void buf_destroy(void *ptr)
{
	struct buf_region *reg = (ptr - 16);

	if (reg->opencl) {
		clSVMFree(reg->opencl->context, reg);
	} else {
		free(reg);
	}
}

static int __buf_map(struct opencl *opencl, void *ptr,
		     size_t size, uint32_t flags)
{
	cl_map_flags cl_flags = 0;

	if (!flags)
		return -EINVAL;

	if (!opencl)
		return 0;

	if (flags & BUF_MAP_WRITE)
		cl_flags |= CL_MAP_WRITE;
	if (flags & BUF_MAP_READ)
		cl_flags |= CL_MAP_READ;

	return clEnqueueSVMMap(opencl->queue, CL_TRUE,
			       cl_flags, ptr, size,
			       0, NULL, NULL);
}

static int buf_map(void *ptr, uint32_t flags)
{
	struct buf_region *reg = (ptr - 16);

	return __buf_map(reg->opencl, ptr, reg->size, flags);
}

static int __buf_unmap(struct opencl *opencl, void *ptr)
{
	if (!opencl)
		return 0;

	return clEnqueueSVMUnmap(opencl->queue, ptr, 0, NULL, NULL);
}

static int buf_unmap(void *ptr)
{
	struct buf_region *reg = (ptr - 16);

	return __buf_unmap(reg->opencl, ptr);
}

static inline unsigned long long nsecs(void)
{
	struct timespec ts = {0, 0};

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((unsigned long long)ts.tv_sec * 1000000000ull) + ts.tv_nsec;
}

static inline float clamp(float lo, float hi, float v)
{
	return MAX(lo, MIN(hi, v));
}

#endif /* !__OPENCL__ */

static inline float deg2rad(float deg)
{
	return deg * M_PI / 180;
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

enum ops_type {
	SPHERE_INTERSECT,
	SPHERE_GET_SURFACE_PROPS,

	TRIANGLE_MESH_INTERSECT,
	TRIANGLE_MESH_GET_SURFACE_PROPS,

	DISTANT_LIGHT_ILLUMINATE,
	POINT_LIGHT_ILLUMINATE,
};

enum material_type {
	MATERIAL_PHONG
};

struct object;

struct object_ops {
	void (*destroy)(struct object *obj);
	int (*unmap)(struct object *obj);
	bool (*intersect)(__global struct object *obj, const vec3_t *orig, const vec3_t *dir,
			  float *near, uint32_t *index, vec2_t *uv);
	void (*get_surface_props)(__global struct object *obj, const vec3_t *hit_point,
				  const vec3_t *dir, uint32_t index, const vec2_t *uv,
				  vec3_t *hit_normal,
				  vec2_t *hit_tex_coords);

	/* Required for OpenCL "virtual" calls */
	enum ops_type intersect_type;
	enum ops_type get_surface_props_type;
};

struct object {
	struct object_ops ops;	 /* because of opencl can't be a pointer */
	struct list_head entry;
	mat4_t o2w;
	mat4_t w2o;
	enum material_type material;
	float albedo;
	float Kd;  /* diffuse weight */
	float Ks;  /* specular weight */
	float n;   /* specular exponent */
};

struct sphere {
	struct object obj;
	float radius;
	float radius_pow2;
	vec3_t center;
};

static bool sphere_intersect(__global struct object *obj, const vec3_t *orig,
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

static void sphere_get_surface_props(__global struct object *obj,
				     const vec3_t *hit_point,
				     const vec3_t *dir, uint32_t index,
				     const vec2_t *uv, vec3_t *hit_normal,
				     vec2_t *hit_tex_coords)
{
	__global struct sphere *sphere;

	sphere = container_of(obj, typeof(*sphere), obj);

	*hit_normal = v3_norm(v3_sub(*hit_point, sphere->center));

	/*
	 * In this particular case, the normal is simular to a point on a unit sphere
	 * centred around the origin. We can thus use the normal coordinates to compute
	 * the spherical coordinates of Phit.
	 * atan2 returns a value in the range [-pi, pi] and we need to remap it to range [0, 1]
	 * acosf returns a value in the range [0, pi] and we also need to remap it to the range [0, 1]
	 */
	hit_tex_coords->x = (1 + atan2(hit_normal->z, hit_normal->x) / M_PI) * 0.5;
	hit_tex_coords->y = acosf(hit_normal->y) / M_PI;
}

struct triangle_mesh {
	struct object	  obj;
	bool		  smooth_shading; /* smooth shading */
	uint32_t	  num_tris;	  /* number of triangles */
	__global vec3_t	  *P;		  /* triangles vertex position */
	__global uint32_t *tris_index;	  /* vertex index array */
	__global vec3_t	  *N;		  /* triangles vertex normals */
	__global vec2_t	  *sts;		  /* triangles texture coordinates */
};

static bool
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

static bool triangle_mesh_intersect(__global struct object *obj, const vec3_t *orig,
				    const vec3_t *dir, float *near,
				    uint32_t *index, vec2_t *uv)
{
	__global struct triangle_mesh *mesh;

	uint32_t j, i;
	bool isect;

	mesh = container_of(obj, typeof(*mesh), obj);

	isect = false;
	for (i = 0, j = 0; i < mesh->num_tris; i++) {
		__global const vec3_t *P = mesh->P;
		__global const vec3_t *v0 = &P[mesh->tris_index[j + 0]];
		__global const vec3_t *v1 = &P[mesh->tris_index[j + 1]];
		__global const vec3_t *v2 = &P[mesh->tris_index[j + 2]];
		float t = INFINITY, u, v;

		if (triangle_intersect(orig, dir, v0, v1, v2, &t, &u, &v) &&
		    t < *near) {
			*near = t;
			uv->x = u;
			uv->y = v;
			*index = i;
			isect = true;
		}
		j += 3;
	}

	return isect;
}

static void triangle_mesh_get_surface_props(__global struct object *obj,
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
		__global const vec3_t *N = mesh->N;
		vec3_t n0 = N[index * 3 + 0];
		vec3_t n1 = N[index * 3 + 1];
		vec3_t n2 = N[index * 3 + 2];

		n0 = v3_muls(n0, 1 - uv->x - uv->y);
		n1 = v3_muls(n1, uv->x);
		n1 = v3_muls(n2, uv->y);

		*hit_normal = v3_add(n2, v3_add(n0, n1));
	} else {
		/* face normal */
		__global const vec3_t *P = mesh->P;
		vec3_t v0 = P[mesh->tris_index[index * 3 + 0]];
		vec3_t v1 = P[mesh->tris_index[index * 3 + 1]];
		vec3_t v2 = P[mesh->tris_index[index * 3 + 2]];

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

struct light;

struct light_ops {
	int (*unmap)(struct light *light);
	void (*illuminate)(__global struct light *light, const vec3_t *orig,
			   vec3_t *dir, vec3_t *intensity, float *distance);

	/* Required for OpenCL "virtual" calls */
	enum ops_type illuminate_type;
};

struct light {
	struct light_ops ops;	/* because of opencl can't a pointer */
	struct list_head entry;
	vec3_t color;
	float intensity;
	mat4_t l2w;
};

struct distant_light {
	struct light light;
	vec3_t dir;
};

static void distant_light_illuminate(__global struct light *light, const vec3_t *orig,
				     vec3_t *dir, vec3_t *intensity, float *distance)
{
	__global struct distant_light *dlight;

	dlight = container_of(light, typeof(*dlight), light);

	*dir = dlight->dir;
	*intensity = v3_muls(dlight->light.color, dlight->light.intensity);
	*distance = INFINITY;
}

struct point_light {
	struct light light;
	vec3_t pos;
};

static void point_light_illuminate(__global struct light *light, const vec3_t *orig,
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
	switch (obj->ops.intersect_type) {
	case SPHERE_INTERSECT:
		return sphere_intersect(obj, orig, dir, near, index, uv);
	case TRIANGLE_MESH_INTERSECT:
		return triangle_mesh_intersect(obj, orig, dir, near, index, uv);
	default:
		/* Hm .. */
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
	switch (obj->ops.get_surface_props_type) {
	case SPHERE_GET_SURFACE_PROPS:
		sphere_get_surface_props(obj, hit_point, dir, index,
					 uv, hit_normal, hit_tex_coords);
		return;
	case TRIANGLE_MESH_GET_SURFACE_PROPS:
		triangle_mesh_get_surface_props(obj, hit_point, dir, index,
						uv, hit_normal, hit_tex_coords);
		return;
	default:
		/* Hm .. */
		return;
	}
#endif
}

static inline void
light_illuminate(__global struct light *light, const vec3_t *orig,
		 vec3_t *dir, vec3_t *intensity, float *distance)
{
#ifndef __OPENCL__
	light->ops.illuminate(light, orig, dir, intensity, distance);
#else
	/* OpenCL does not support function pointers, se la vie	 */
	switch (light->ops.illuminate_type) {
	case DISTANT_LIGHT_ILLUMINATE:
		distant_light_illuminate(light, orig, dir, intensity, distance);
		return;
	case POINT_LIGHT_ILLUMINATE:
		point_light_illuminate(light, orig, dir, intensity, distance);
		return;
	default:
		/* Hm .. */
		return;
	}
#endif
}

enum ray_type {
	PRIMARY_RAY,
	SHADOW_RAY
};

struct intersection {
	__global struct object *hit_object;
	float near;
	vec2_t uv;
	uint32_t index;
};

static bool trace(__global struct scene *scene, const vec3_t *orig, const vec3_t *dir,
		  struct intersection *isect, enum ray_type ray_type)
{
	__global struct object *obj;

	isect->hit_object = NULL;
	isect->near = INFINITY;

	list_for_each_entry(obj, &scene->objects, entry) {
		float near = INFINITY;
		uint32_t index = 0;
		vec2_t uv;

		if (object_intersect(obj, orig, dir, &near, &index, &uv) &&
		    near < isect->near) {
			isect->hit_object = obj;
			isect->near = near;
			isect->index = index;
			isect->uv = uv;
		}
	}

	return !!isect->hit_object;
}

/**
 * Compute reflection direction
 */
static vec3_t reflect(const vec3_t *I, const vec3_t *N)
{
	float dot = v3_dot(*I, *N);

	return v3_sub(*I, v3_muls(*N, 2 * dot));
}

/**
 * Compute refraction direction
 */
__attribute__((unused))
static vec3_t refract(const vec3_t *I, const vec3_t *N, float ior)
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
__attribute__((unused))
static void fresnel(const vec3_t *I, const vec3_t *N, float ior, float *kr)
{
	float cosi = clamp(-1.0f, 1.0f, v3_dot(*I, *N));
	float etai = 1, etat = ior;
	if (cosi > 0)
		SWAP(etai, etat);

	/* Compute sini using Snell's law */
	float sint = etai / etat * sqrtf(MAX(0.0f, 1.0f - cosi * cosi));

	/* Total internal reflection */
	if (sint >= 1) {
		*kr = 1;
	} else {
		float cost, Rs, Rp;

		cost = sqrtf(MAX(0.0f, 1.0f - sint * sint));
		cosi = fabsf(cosi);
		Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		*kr = (Rs * Rs + Rp * Rp) / 2;
	}
	/*
	 * As a consequence of the conservation of energy,
	 * transmittance is given by:
	 * kt = 1 - kr;
	 */
}

static vec3_t ray_cast(__global struct scene *scene, const vec3_t *orig,
		       const vec3_t *dir, uint32_t depth)
{
	struct intersection isect;
	vec3_t hit_color;

	if (depth > scene->max_depth)
		return scene->back_color;

	if (trace(scene, orig, dir, &isect, PRIMARY_RAY)) {
		/* Evaluate surface properties (P, N, texture coordinates, etc.) */

		vec3_t hit_point;
		vec3_t hit_normal;
		vec2_t hit_tex_coords;

		hit_point = v3_add(v3_muls(*dir, isect.near), *orig);
		object_get_surface_props(isect.hit_object, &hit_point, dir,
					 isect.index, &isect.uv,
					 &hit_normal, &hit_tex_coords);
		switch (isect.hit_object->material) {
		case MATERIAL_PHONG: {
			/*
			 * Light loop (loop over all lights in the scene
			 * and accumulate their contribution)
			 */
			vec3_t diffuse, specular;
			__global struct light *light;

			diffuse = specular = vec3(0.0f, 0.0f, 0.0f);

			list_for_each_entry(light, &scene->lights, entry) {
				vec3_t light_dir, light_intensity;
				vec3_t point, rev_light_dir, R;
				vec3_t rev_dir, diff, spec;

				struct intersection isect_shadow;
				float near, p;
				bool obstacle;

				light_illuminate(light, &hit_point, &light_dir,
						 &light_intensity, &near);

				point = v3_add(hit_point, v3_muls(hit_normal, scene->bias));
				rev_light_dir = v3_muls(light_dir, -1.0f);

				obstacle = !!trace(scene, &point, &rev_light_dir,
						   &isect_shadow, SHADOW_RAY);
				if (obstacle)
					/* Light is not visible, object is hit, thus shadow */
					continue;

				/* compute the diffuse component */
				diff = v3_muls(light_intensity, isect.hit_object->albedo *
					       MAX(0.0f, v3_dot(hit_normal, rev_light_dir)));
				diffuse = v3_add(diffuse, diff);

				/*
				 * compute the specular component
				 * what would be the ideal reflection direction for this
				 * light ray
				 */
				R = reflect(&light_dir, &hit_normal);

				rev_dir = v3_muls(*dir, -1.0f);

				p = powf(MAX(0.0f, v3_dot(R, rev_dir)), isect.hit_object->n);
				spec = v3_muls(light_intensity, p);
				specular = v3_add(specular, spec);
			}
			/* Compute the whole light contribution */
			diffuse = v3_muls(diffuse, isect.hit_object->Kd);
			specular = v3_muls(specular, isect.hit_object->Ks);
			hit_color = v3_add(diffuse, specular);
			break;
		}
		default:
			hit_color = vec3(0.0f, 0.0f, 0.0f);
			break;
		}
	} else {
		hit_color = scene->back_color;
	}

	return hit_color;
}

static inline void color_vec_to_rgba32(const vec3_t *color, struct rgba *rgb)
{
	*rgb = (struct rgba) {
		.r = (255 * clamp(0.0f, 1.0f, color->x)),
		.g = (255 * clamp(0.0f, 1.0f, color->y)),
		.b = (255 * clamp(0.0f, 1.0f, color->z))
	};
}

#ifdef __OPENCL__

__kernel void render_opencl(__global struct scene *scene)
{
	float x, y, scale, img_ratio;
	vec3_t orig, dir, color;
	uint32_t i, ix, iy;

	scale = tan(deg2rad(scene->fov * 0.5f));
	img_ratio = scene->width / (float)scene->height;

	/* Camera position */
	orig = m4_mul_pos(scene->c2w, vec3(0.f, 0.f, 0.f));

	i = get_global_id(0);
	iy = i / scene->width;
	ix = i % scene->width;

	x = (2.0f * (ix + 0.5f) / (float)scene->width - 1.0f) * img_ratio * scale;
	y = (1.0f - 2.0f * (iy + 0.5f) / (float)scene->height) * scale;

	dir = m4_mul_dir(scene->c2w, vec3(x, y, -1.0f));
	dir = v3_norm(dir);

	color = ray_cast(scene, &orig, &dir, 0);
	color_vec_to_rgba32(&color, &scene->framebuffer[i]);
}

#else /* !__OPENCL__ */

static int opencl_init(struct opencl *opencl, const char *kernel_fn)
{
	/* Load current file */
	const char *path = __FILE__;

	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_device_svm_capabilities caps;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_command_queue queue;
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_int ret;

	size_t size, read;
	char *source;
	FILE *fp;

	fp = fopen(path, "r");
	assert(fp);

	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	source = malloc(size);
	assert(source);

	read = fread(source, 1, size, fp);
	assert(read == size);

	/* Get platform and device information */
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	assert(!ret);

	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,
			     1, &device_id, &ret_num_devices);
	assert(!ret);

	/* Create an OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	assert(!ret);

	/* Get caps */
	ret = clGetDeviceInfo(device_id, CL_DEVICE_SVM_CAPABILITIES,
			      sizeof(caps), &caps, 0);
	assert(!ret);

	if (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
		/* TODO: support fine grained buffer, map-free */
	}

	/* Create a command queue */
	queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
	assert(!ret);

	/* Create a program from the kernel source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source,
					    &size, &ret);
	assert(!ret);

	free(source);

	/* Build the program */
	ret = clBuildProgram(program, 1, &device_id, "-cl-std=CL2.0 -Werror -D__OPENCL__",
			     NULL, NULL);
	if (ret == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char *log;

		/* Determine the size of the log */
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
				      0, NULL, &log_size);

		/* Allocate memory for the log */
		log = malloc(log_size);
		assert(log);

		/* Get the log */
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
				      log_size, log, NULL);

		/* Print and free the log */
		fprintf(stderr, "%s\n", log);
		free(log);

		return -1;
	} else if (ret) {
		fprintf(stderr, "clBuildProgram: failed %d\n", ret);
		return -1;
	}

	/* Create the opencl kernel */
	kernel = clCreateKernel(program, kernel_fn, &ret);
	assert(!ret);

	/* Init context */
	opencl->context = context;
	opencl->device_id = device_id;
	opencl->queue = queue;
	opencl->program = program;
	opencl->kernel = kernel;

	return 0;
}

static void opencl_deinit(struct opencl *opencl)
{
	if (!opencl)
		return;

	clReleaseKernel(opencl->kernel);
	clReleaseProgram(opencl->program);
	clReleaseCommandQueue(opencl->queue);
	clReleaseContext(opencl->context);
}

static void opencl_invoke(struct scene *scene)
{
	struct opencl *opencl = scene->opencl;
	size_t global_item_size = scene->width * scene->height;
	/* Divide work items into groups of 64 */
	size_t local_item_size = 64;
	int ret;

	ret = clSetKernelArgSVMPointer(opencl->kernel, 0, scene);
	assert(!ret);

	ret = clEnqueueNDRangeKernel(opencl->queue, opencl->kernel, 1, NULL,
				     &global_item_size,
				     &local_item_size, 0,
				     NULL, NULL);
	assert(!ret);
}

static void object_init(struct object *obj, struct object_ops *ops,
			const mat4_t *o2w)
{
	INIT_LIST_HEAD(&obj->entry);
	obj->ops = *ops;
	obj->o2w = *o2w;
	obj->w2o = m4_invert_affine(obj->o2w);
	obj->material = MATERIAL_PHONG;
	obj->albedo = 0.18f;
	obj->Kd = 0.8f;
	obj->Ks = 0.2f;
	obj->n = 10.0f;
}

static void object_destroy(struct object *obj)
{
	list_del(&obj->entry);
	obj->ops.destroy(obj);
}

static void sphere_destroy(struct object *obj)
{
	struct sphere *sphere =
		container_of(obj, struct sphere, obj);

	buf_destroy(sphere);
}

static int sphere_unmap(struct object *obj)
{
	struct sphere *sphere =
		container_of(obj, struct sphere, obj);

	return buf_unmap(sphere);
}

struct object_ops sphere_ops = {
	.destroy		= sphere_destroy,
	.unmap			= sphere_unmap,
	.intersect		= sphere_intersect,
	.intersect_type		= SPHERE_INTERSECT,
	.get_surface_props	= sphere_get_surface_props,
	.get_surface_props_type = SPHERE_GET_SURFACE_PROPS,
};

static void sphere_init(struct sphere *sphere, const mat4_t *o2w, float radius)
{
	object_init(&sphere->obj, &sphere_ops, o2w);

	sphere->radius = radius;
	sphere->radius_pow2 = radius * radius;
	sphere->center = m4_mul_pos(*o2w, vec3(0.0f, 0.0f, 0.0f));
}

static void triangle_mesh_destroy(struct object *obj)
{
	struct triangle_mesh *mesh =
		container_of(obj, struct triangle_mesh, obj);

	buf_destroy(mesh->P);
	buf_destroy(mesh->tris_index);
	buf_destroy(mesh->N);
	buf_destroy(mesh->sts);
	buf_destroy(mesh);
}

static int triangle_mesh_unmap(struct object *obj)
{
	struct triangle_mesh *mesh =
		container_of(obj, struct triangle_mesh, obj);

	return buf_unmap(mesh);
}

struct object_ops triangle_mesh_ops = {
	.destroy		= triangle_mesh_destroy,
	.unmap			= triangle_mesh_unmap,
	.intersect		= triangle_mesh_intersect,
	.intersect_type		= TRIANGLE_MESH_INTERSECT,
	.get_surface_props	= triangle_mesh_get_surface_props,
	.get_surface_props_type = TRIANGLE_MESH_GET_SURFACE_PROPS,

};

static void triangle_mesh_init(struct scene *scene, struct triangle_mesh *mesh,
			       const mat4_t *o2w, uint32_t nfaces, uint32_t *face_index,
			       uint32_t *verts_index, vec3_t *verts,
			       vec3_t *normals, vec2_t *st)
{
	uint32_t i, j, l, k = 0, max_vert_index = 0;
	uint32_t num_tris = 0;

	uint32_t *tris_index;
	vec3_t *P, *N;
	vec2_t *sts;

	mat4_t transform_normals;

	/* find out how many triangles we need to create for this mesh */
	for (i = 0; i < nfaces; ++i) {
		num_tris += face_index[i] - 2;
		for (j = 0; j < face_index[i]; ++j) {
			if (verts_index[k + j] > max_vert_index)
				max_vert_index = verts_index[k + j];
		}
		k += face_index[i];
	}
	max_vert_index += 1;

	/* allocate memory to store the position of the mesh vertices */
	P = buf_allocate(scene->opencl, max_vert_index * sizeof(*P));
	assert(P);

	/* Transform vertices to world space */
	for (i = 0; i < max_vert_index; ++i)
		P[i] = m4_mul_pos(*o2w, verts[i]);

	tris_index = buf_allocate(scene->opencl, num_tris * 3 * sizeof(*tris_index));
	assert(tris_index);

	N = buf_allocate(scene->opencl, num_tris * 3 * sizeof(*N));
	assert(N);

	sts = buf_allocate(scene->opencl, num_tris * 3 * sizeof(*sts));
	assert(sts);

	/* Init object */
	object_init(&mesh->obj, &triangle_mesh_ops, o2w);

	/* Computing the transpse of the object-to-world inverse matrix */
	transform_normals = m4_transpose(mesh->obj.w2o);

	/* Generate the triangle index array and set normals and st coordinates */
	for (i = 0, k = 0, l = 0; i < nfaces; i++) {
		/* Each triangle in a face */
		for (j = 0; j < face_index[i] - 2; j++) {
			tris_index[l + 0] = verts_index[k];
			tris_index[l + 1] = verts_index[k + j + 1];
			tris_index[l + 2] = verts_index[k + j + 2];

			/* Transforming normals */
			N[l + 0] = m4_mul_dir(transform_normals, normals[k]);
			N[l + 1] = m4_mul_dir(transform_normals, normals[k + j + 1]);
			N[l + 2] = m4_mul_dir(transform_normals, normals[k + j + 2]);

			N[l + 0] = v3_norm(N[l + 0]);
			N[l + 1] = v3_norm(N[l + 1]);
			N[l + 2] = v3_norm(N[l + 2]);

			sts[l + 0] = st[k];
			sts[l + 1] = st[k + j + 1];
			sts[l + 2] = st[k + j + 2];
			l += 3;
		}
		k += face_index[i];
	}

	mesh->num_tris = num_tris;
	mesh->P = P;
	mesh->tris_index = tris_index;
	mesh->N = N;
	mesh->sts = sts;

	/* Not supposed to changed by the host, so unmap */
	buf_unmap(P);
	buf_unmap(tris_index);
	buf_unmap(N);
	buf_unmap(sts);
}

static int mesh_load(struct scene *scene, const char *file,
		     const mat4_t *o2w)
{
	uint32_t num_faces, verts_ind_arr_sz, verts_arr_sz;
	int ret, i;
	FILE *f;

	uint32_t *face_index, *verts_index;
	vec3_t *verts, *normals;
	vec2_t *st;

	struct triangle_mesh *mesh;

	mesh = buf_allocate(scene->opencl, sizeof(*mesh));
	assert(mesh);

	f = fopen(file, "r");
	if (!f) {
		fprintf(stderr, "Can't open file: %s\n", file);
		return -EINVAL;
	}

	ret = fscanf(f, "%d", &num_faces);
	assert(ret == 1);

	face_index = calloc(num_faces, sizeof(*face_index));
	assert(face_index);

	for (i = 0, verts_ind_arr_sz = 0; i < num_faces; i++) {
		ret = fscanf(f, "%d", &face_index[i]);
		assert(ret == 1);
		verts_ind_arr_sz += face_index[i];
	}

	verts_index = calloc(verts_ind_arr_sz, sizeof(*verts_index));
	assert(verts_index);

	for (i = 0, verts_arr_sz = 0; i < verts_ind_arr_sz; i++) {
		ret = fscanf(f, "%d", &verts_index[i]);
		assert(ret == 1);
		if (verts_index[i] > verts_arr_sz)
			verts_arr_sz = verts_index[i];
	}
	verts_arr_sz += 1;

	verts = calloc(verts_arr_sz, sizeof(*verts));
	assert(verts);

	for (i = 0; i < verts_arr_sz; i++) {
		vec3_t *vert = &verts[i];
		ret = fscanf(f, "%f%f%f", &vert->x, &vert->y, &vert->z);
		assert(ret == 3);
	}

	normals = calloc(verts_ind_arr_sz, sizeof(*normals));
	assert(normals);

	for (i = 0; i < verts_ind_arr_sz; i++) {
		vec3_t *norm = &normals[i];
		ret = fscanf(f, "%f%f%f", &norm->x, &norm->y, &norm->z);
		assert(ret == 3);
	}

	st = calloc(verts_ind_arr_sz, sizeof(*st));
	assert(st);

	for (i = 0; i < verts_ind_arr_sz; i++) {
		vec2_t *coord = &st[i];
		ret = fscanf(f, "%f%f", &coord->x, &coord->y);
		assert(ret == 2);
	}
	fclose(f);

	triangle_mesh_init(scene, mesh, o2w, num_faces, face_index,
			   verts_index, verts, normals, st);

	free(face_index);
	free(verts_index);
	free(verts);
	free(normals);
	free(st);

	list_add_tail(&mesh->obj.entry, &scene->objects);

	return 0;
}

static int objects_create(struct scene *scene)
{
	float w[5] = {0.04, 0.08, 0.1, 0.15, 0.2};
	int ret, i = -4, n = 2, k = 0;

	mat4_t o2w;

	/* Init mesh */
	o2w = m4_identity();
	ret = mesh_load(scene, "./plane.geo", &o2w);
	if (ret)
		return -ENOMEM;

	/* Init objects */
	for (; i <= 4; i+= 2, n *= 5, k++) {
		struct sphere *sphere;

		sphere = buf_allocate(scene->opencl, sizeof(*sphere));
		if (!sphere)
			return -ENOMEM;

		/* Object position */
		o2w = m4_identity();
		o2w.m30 = i;
		o2w.m31 = 1;

		sphere_init(sphere, &o2w, 0.9);
		sphere->obj.n = n;
		sphere->obj.Ks = w[k];

		list_add_tail(&sphere->obj.entry, &scene->objects);
	}

	return 0;
}

static void objects_destroy(struct scene *scene)
{
	struct object *obj, *tmp;

	list_for_each_entry_safe(obj, tmp, &scene->objects, entry)
		object_destroy(obj);
}

static void light_init(struct light *light, struct light_ops *ops,
		       const mat4_t *l2w, const vec3_t *color, float intensity)
{
	INIT_LIST_HEAD(&light->entry);
	light->ops = *ops;
	light->l2w = *l2w;
	light->color = *color;
	light->intensity = intensity;
}

static int distant_light_unmap(struct light *light)
{
	struct distant_light *dlight =
		container_of(light, struct distant_light, light);

	return buf_unmap(dlight);
}

struct light_ops distant_light_ops = {
	.unmap		 = distant_light_unmap,
	.illuminate	 = distant_light_illuminate,
	.illuminate_type = DISTANT_LIGHT_ILLUMINATE,
};

static void distant_light_init(struct distant_light *dlight, const mat4_t *l2w,
			       const vec3_t *color, float intensity)
{
	vec3_t dir;

	light_init(&dlight->light, &distant_light_ops, l2w, color, intensity);

	dir = m4_mul_dir(*l2w, vec3(0.0f, 0.0f, -1.0f));
	/* in case the matrix scales the light */
	dlight->dir = v3_norm(dir);
}

static int point_light_unmap(struct light *light)
{
	struct point_light *plight =
		container_of(light, struct point_light, light);

	return buf_unmap(plight);
}

struct light_ops point_light_ops = {
	.unmap		 = point_light_unmap,
	.illuminate	 = point_light_illuminate,
	.illuminate_type = POINT_LIGHT_ILLUMINATE,
};

__attribute__((unused))
static void point_light_init(struct point_light *plight, const mat4_t *l2w,
			     const vec3_t *color, float intensity)
{
	light_init(&plight->light, &point_light_ops, l2w, color, intensity);
	plight->pos = m4_mul_pos(*l2w, vec3(0.0f, 0.0f, 0.0f));
}

static int lights_create(struct scene *scene)
{
	struct distant_light *dlight;

	mat4_t l2w;
	vec3_t color;
	float intensity;

	l2w = mat4(11.146836, -5.781569, -0.0605886, 0,
		   -1.902827, -3.543982, -11.895445, 0,
		   5.459804, 10.568624, -4.02205, 0,
		   0, 0, 0, 1);
	/* Row-major to column-major */
	l2w = m4_transpose(l2w);

	color = vec3(1.f, 1.f, 1.f);
	intensity = 5;

	dlight = buf_allocate(scene->opencl, sizeof(*dlight));
	if (!dlight)
		return -ENOMEM;

	distant_light_init(dlight, &l2w, &color, intensity);
	list_add_tail(&dlight->light.entry, &scene->lights);

	return 0;
}

static void lights_destroy(struct scene *scene)
{
	struct light *light, *tmp;

	list_for_each_entry_safe(light, tmp, &scene->lights, entry) {
		list_del(&light->entry);
		buf_destroy(light);
	}
}

static int sdl_init(struct scene *scene)
{
	struct sdl *sdl;
	int ret;

	ret = SDL_Init(SDL_INIT_VIDEO);
	if (ret) {
		printf("Can't init SDL\n");
		return -1;
	}

	sdl = malloc(sizeof(*sdl));
	assert(sdl);

	sdl->window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED,
				       SDL_WINDOWPOS_CENTERED,
				       scene->width, scene->height,
				       SDL_WINDOW_RESIZABLE);
	assert(sdl->window);

	sdl->renderer = SDL_CreateRenderer(sdl->window, -1,
					   SDL_RENDERER_PRESENTVSYNC);
	assert(sdl->renderer);

	SDL_SetWindowMinimumSize(sdl->window, scene->width, scene->height);
	SDL_RenderSetLogicalSize(sdl->renderer, scene->width, scene->height);
	SDL_RenderSetIntegerScale(sdl->renderer, SDL_TRUE);
	SDL_SetRenderDrawBlendMode(sdl->renderer, SDL_BLENDMODE_BLEND);

	sdl->screen = SDL_CreateTexture(sdl->renderer, SDL_PIXELFORMAT_RGBA8888,
					SDL_TEXTUREACCESS_STREAMING,
					scene->width, scene->height);
	assert(sdl->screen);

	scene->sdl = sdl;

	return 0;
}

static void sdl_deinit(struct scene *scene)
{
	struct sdl *sdl = scene->sdl;

	if (!sdl)
		return;

	SDL_DestroyTexture(sdl->screen);
	SDL_DestroyRenderer(sdl->renderer);
	SDL_DestroyWindow(sdl->window);
	free(sdl);
	scene->sdl = NULL;
}

static void camera_set_angles(struct scene *scene, float pitch, float yaw)
{
	struct camera *cam = &scene->cam;

	if (pitch >= 90.0)
		pitch = 89.9;
	else if (pitch <= -90.0)
		pitch = -89.9;

	// -Z axis (0, 0, -1): pitch -> yaw
	cam->dir.x = sin(deg2rad(yaw))*cos(deg2rad(pitch));
	cam->dir.y = sin(deg2rad(pitch));
	cam->dir.z = -cos(deg2rad(yaw))*cos(deg2rad(pitch));

	cam->pitch = pitch;
	cam->yaw = yaw;
}

static void camera_update_c2w(struct scene *scene)
{
	struct camera *cam = &scene->cam;
	int ret;

	ret = __buf_map(scene->opencl, &scene->c2w, sizeof(scene->c2w),
			BUF_MAP_WRITE);
	assert(!ret);

	scene->c2w = m4_look_at(cam->pos, v3_add(cam->pos, cam->dir),
				vec3(0.0f, 1.0f, 0.0f));

	ret = __buf_unmap(scene->opencl, &scene->c2w);
	assert(!ret);
}

static void camera_inc_angles(struct scene *scene, float inc_pitch, float inc_yaw)
{
	struct camera *cam = &scene->cam;
	camera_set_angles(scene, cam->pitch + inc_pitch, cam->yaw + inc_yaw);
}

static struct scene *scene_create(struct opencl *opencl, bool no_sdl,
				  uint32_t width, uint32_t height,
				  vec3_t cam_pos, float cam_pitch,
				  float cam_yaw, float fov)
{
	struct scene *scene;
	struct rgba *framebuffer;
	int ret;

	/* Don't mmap by default */
	framebuffer = __buf_allocate(opencl, width * height * sizeof(*framebuffer), 0);
	assert(framebuffer);

	scene = buf_allocate(opencl, sizeof(*scene));
	assert(scene);

	*scene = (struct scene) {
		.width	      = width,
		.height	      = height,
		.fov	      = fov,
		.back_color   = {0.235294f, 0.67451f, 0.843137f},
		.c2w	      = m4_identity(),
		.bias	      = 0.0001,
		.opencl	      = opencl,
		.max_depth    = 5,
		.framebuffer  = framebuffer,
		.objects      = LIST_HEAD_INIT(scene->objects),
		.lights	      = LIST_HEAD_INIT(scene->lights),

		.cam = {
			.pos   = cam_pos,
		},
	};
	camera_set_angles(scene, cam_pitch, cam_yaw);
	camera_update_c2w(scene);

	if (!no_sdl) {
		ret = sdl_init(scene);
		assert(!ret);
	}

	return scene;
};

static void scene_destroy(struct scene *scene)
{
	sdl_deinit(scene);
	objects_destroy(scene);
	lights_destroy(scene);
	buf_destroy(scene->framebuffer);
	buf_destroy(scene);
}

static int scene_finish(struct scene *scene)
{
	struct object *object;
	struct light *light;
	int ret;

	list_for_each_entry(object, &scene->objects, entry) {
		ret = object->ops.unmap(object);
		if (ret)
			return ret;
	}

	list_for_each_entry(light, &scene->lights, entry) {
		ret = light->ops.unmap(light);
		if (ret)
			return ret;
	}

	return buf_unmap(scene);
}

static void render_soft(struct scene *scene)
{
	float scale, img_ratio;
	vec3_t orig, color;
	struct rgba *pix;
	uint32_t i, j;

	scale = tan(deg2rad(scene->fov * 0.5));
	img_ratio = scene->width / (float)scene->height;

	/* Camera position */
	orig = m4_mul_pos(scene->c2w, vec3(0.f, 0.f, 0.f));

	pix = scene->framebuffer;
	for (j = 0; j < scene->height; ++j) {
		for (i = 0; i < scene->width; ++i) {
			float x = (2 * (i + 0.5) / (float)scene->width - 1) *	img_ratio * scale;
			float y = (1 - 2 * (j + 0.5) / (float)scene->height) * scale;
			vec3_t dir;

			dir = m4_mul_dir(scene->c2w, vec3(x, y, -1));
			dir = v3_norm(dir);

			color = ray_cast(scene, &orig, &dir, 0);
			color_vec_to_rgba32(&color, pix);
			pix++;
		}
	}
}

static void one_frame_render(struct scene *scene)
{
	unsigned long long ns;
	FILE *out;
	int i, ret;

	ns = nsecs();
	if (scene->opencl) {
		opencl_invoke(scene);
	} else {
		render_soft(scene);
	}
	fprintf(stderr, "\rDone: %.6f (sec)\n", (nsecs() - ns) / 1000000000.0);

	/* save framebuffer to file */
	out = fopen("yart-out.ppm", "w");
	assert(out);

	/* Map for reading */
	ret = buf_map(scene->framebuffer, BUF_MAP_READ);
	assert(!ret);

	fprintf(out, "P6\n%d %d\n255\n", scene->width, scene->height);
	for (i = 0; i < scene->height * scene->width; ++i) {
		struct rgba *rgb = &scene->framebuffer[i];

		fprintf(out, "%c%c%c", rgb->r, rgb->g, rgb->b);
	}
	fclose(out);

	/* Unmap */
	ret = buf_unmap(scene->framebuffer);
	assert(!ret);
}

static void render(struct scene *scene)
{
	struct sdl *sdl = scene->sdl;
	unsigned long long ns, fps;
	int ret;

	SDL_SetRelativeMouseMode(SDL_TRUE);
	SDL_StopTextInput();

	ns = nsecs();
	fps = 0;

	/* Main render loop */
	while (1) {
		struct camera *cam = &scene->cam;
		SDL_Event event;
		SDL_Point mouse;
		const uint8_t *keyb;
		bool updated_cam = false;

		SDL_GetRelativeMouseState(&mouse.x, &mouse.y);
		keyb = SDL_GetKeyboardState(NULL);

		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT)
				/* Exit */
				return;

			if (event.type == SDL_KEYDOWN) {
				if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
					/* Exit */
					return;
			}
		}

		/* Handle mouse movement */
		if (mouse.y && mouse.x) {
			camera_inc_angles(scene, -mouse.y * 0.05f, mouse.x * 0.05f);
			updated_cam = true;
		}

		/* Handle keyboard */
		if (keyb[SDL_SCANCODE_W]) {
			cam->pos = v3_add(cam->pos, v3_muls(cam->dir, 0.1f));
			updated_cam = true;
		}
		else if (keyb[SDL_SCANCODE_S]) {
			cam->pos = v3_sub(cam->pos, v3_muls(cam->dir, 0.1f));
			updated_cam = true;
		}
		if (keyb[SDL_SCANCODE_A]) {
			vec3_t up = vec3(0.0f, 1.0f, 0.0f);
			vec3_t right = v3_cross(cam->dir, up);

			cam->pos = v3_sub(cam->pos, v3_muls(right, 0.1f));
			updated_cam = true;
		}
		else if (keyb[SDL_SCANCODE_D]) {
			vec3_t up = vec3(0.0f, 1.0f, 0.0f);
			vec3_t right = v3_cross(cam->dir, up);

			cam->pos = v3_add(cam->pos, v3_muls(right, 0.1f));
			updated_cam = true;
		}

		/* Update cam-to-world matrix */
		if (updated_cam)
			camera_update_c2w(scene);

		/* Render one frame */
		if (scene->opencl) {
			opencl_invoke(scene);
		} else {
			render_soft(scene);
		}

		/* Map for reading */
		ret = buf_map(scene->framebuffer, BUF_MAP_READ);
		assert(!ret);

		SDL_RenderClear(sdl->renderer);
		SDL_UpdateTexture(sdl->screen, NULL, scene->framebuffer,
				  scene->width * sizeof(*scene->framebuffer));
		SDL_RenderCopy(sdl->renderer, sdl->screen, NULL, NULL);
		SDL_RenderPresent(sdl->renderer);

		/* Unmap */
		ret = buf_unmap(scene->framebuffer);
		assert(!ret);

		fps++;
		if (nsecs() - ns >= 1000000000ull) {
			fprintf(stderr, "\r%lld FPS", fps);
			ns = nsecs();
			fps = 0;
		}
	}
}

static int no_opencl;
static int one_frame;

enum {
	OPT_FOV = 'a',
	OPT_SCREEN_WIDTH,
	OPT_SCREEN_HEIGHT,

	OPT_CAM_PITCH,
	OPT_CAM_YAW,
	OPT_CAM_POS,
};

static struct option long_options[] = {
	{"no-opencl", no_argument,       &no_opencl, 1},
	{"opencl",    no_argument,       &no_opencl, 0},
	{"one-frame", no_argument,       &one_frame, 1},
	{"fov",       required_argument, 0, OPT_FOV},
	{"width",     required_argument, 0, OPT_SCREEN_WIDTH},
	{"height",    required_argument, 0, OPT_SCREEN_HEIGHT},
	{"pitch",     required_argument, 0, OPT_CAM_PITCH},
	{"yaw",       required_argument, 0, OPT_CAM_YAW},
	{"pos",       required_argument, 0, OPT_CAM_POS},

	{0, 0, 0, 0}
};

static void usage(void)
{
	printf("Usage:\n"
	       "  $ yart [--no-opencl] [--one-frame] [--fov <fov>] [--width <width>] [--height <height>]\n"
	       "         [--pitch <pitch>] [--yaw <yaw>] [--pos <pos>]"
	       "\n"
	       "OPTIONS:\n"
	       "   --no-opencl  - no OpenCL hardware accelaration\n"
	       "   --one-frame  - render one frame and exit\n"
	       "\n"
	       "ARGUMENTS:\n"
	       "   --fov       - field of view angle in degrees (float)\n"
	       "   --width     - screen width (integer)\n"
	       "   --height    - screen height (integer)\n"
	       "   --pitch     - initial camera pitch angle in degrees (float)\n"
	       "   --yaw       - initial camera yaw angle in degrees (float)\n"
	       "   --pos       - initial camera position in format x,y,z.\n"
	       "                 e.g: '--pos 0.0,1.0,12.0'\n"
	       "\n"
		);

	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	struct opencl __opencl, *opencl = NULL;
	struct scene *scene;

	uint32_t width = 1024;
	uint32_t height = 768;

	float cam_pitch = 0.0f;
	float cam_yaw = 0.0f;
	vec3_t cam_pos = vec3(0.0f, 2.0f, 16.0f);
	float fov = 27.95f; /* 50mm focal lengh */

	int ret;

	while (1) {
		int c, ret, option_index = 0;

		c = getopt_long(argc, argv, "", long_options, &option_index);
		if (c == -1)
			break;

		switch (c) {
		case 0:
			break;
		case OPT_FOV:
			ret = sscanf(optarg, "%f", &fov);
			if (ret != 1) {
				fprintf(stderr, "Invalid --fov, should be float.\n");
				exit(EXIT_FAILURE);
			}
			break;
		case OPT_SCREEN_WIDTH:
			ret = sscanf(optarg, "%u", &width);
			if (ret != 1) {
				fprintf(stderr, "Invalid --width, should be integer.\n");
				exit(EXIT_FAILURE);
			}
			break;
		case OPT_SCREEN_HEIGHT:
			ret = sscanf(optarg, "%u", &height);
			if (ret != 1) {
				fprintf(stderr, "Invalid --height, should be integer.\n");
				exit(EXIT_FAILURE);
			}
			break;
		case OPT_CAM_PITCH:
			ret = sscanf(optarg, "%f", &cam_pitch);
			if (ret != 1) {
				fprintf(stderr, "Invalid --camera-pitch, should be float.\n");
				exit(EXIT_FAILURE);
			}
			break;
		case OPT_CAM_YAW:
			ret = sscanf(optarg, "%f", &cam_yaw);
			if (ret != 1) {
				fprintf(stderr, "Invalid --camera-yaw, should be float.\n");
				exit(EXIT_FAILURE);
			}
			break;
		case OPT_CAM_POS:
			ret = sscanf(optarg, "%f,%f,%f", &cam_pos.x, &cam_pos.y, &cam_pos.z);
			if (ret != 3) {
				fprintf(stderr, "Invalid --camera-pos, should be float,float,float.\n");
				exit(EXIT_FAILURE);
			}
			break;

		case '?':
			usage();
			break;
		default:
			usage();
		}
	}

	if (!no_opencl) {
		/* Init opencl context */
		opencl = &__opencl;
		ret = opencl_init(opencl, "render_opencl");
		if (ret)
			return -1;
	}

	/* Create scene */
	scene = scene_create(opencl, one_frame, width, height,
			     cam_pos, cam_pitch, cam_yaw, fov);
	assert(scene);

	/* Init default objects */
	ret = objects_create(scene);
	assert(!ret);

	/* Init default light */
	ret = lights_create(scene);
	assert(!ret);

	/* Commit all scene changes before rendering */
	ret = scene_finish(scene);
	assert(!ret);

	if (one_frame)
		one_frame_render(scene);
	else
		render(scene);

	scene_destroy(scene);
	opencl_deinit(opencl);

	return 0;
}

#endif /* !__OPENCL__ */
