// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * YART (Yet Another Ray Tracer) boosted by OpenCL
 * Copyright (C) 2020,2021 Roman Penyaev
 *
 * Based on lessons from scratchapixel.com and pbr-book.org
 *
 * Roman Penyaev <r.peniaev@gmail.com>
 */

#ifndef __OPENCL__
#define _GNU_SOURCE
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
#include <SDL2/SDL_ttf.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/vector3.h>

#define __global

#else /* __OPENCL__ */

#define sinf  sin
#define cosf  cos
#define tanf  tan
#define acosf acos
#define fabsf fabs
#define sqrtf sqrt
#define powf  pow
#define floorf floor

struct scene;

typedef unsigned long uint64_t;
typedef unsigned int  uint32_t;
typedef int	      int32_t;

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

#define EPSILON	   1e-8
#define MOVE_SPEED 0.03f

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
	vec3_t	 backcolor;
	mat4_t	 c2w;
	float	 bias;
	uint32_t ray_depth;
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
	struct buf_region *reg;

	if (!ptr)
		return;

	reg = (ptr - 16);
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
	MATERIAL_PHONG,
	MATERIAL_DIFFUSE,
	MATERIAL_REFLECT,
	MATERIAL_REFLECT_REFRACT,
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
	enum material_type material;
	float albedo;
	float ior; /* index of refraction */
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
	 * In this particular case, the normal is similar to a point on a unit sphere
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
	uint32_t	  num_verts;	  /* number of vertices */
	__global vec3_t	  *vertices;	  /* vertex positions */
	__global vec3_t	  *normals;	  /* vertex normals */
	__global vec2_t	  *sts;		  /* texture coordinates */
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

	uint32_t i;
	bool isect;

	mesh = container_of(obj, typeof(*mesh), obj);

	isect = false;
	for (i = 0; i < mesh->num_verts; i += 3) {
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
			*index = i / 3;
			isect = true;
		}
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

struct light;

struct light_ops {
	void (*destroy)(struct light *light);
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
			if (ray_type == SHADOW_RAY &&
			    obj->material == MATERIAL_REFLECT_REFRACT)
				continue;
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
static float fresnel(const vec3_t *I, const vec3_t *N, float ior)
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

static inline float modulo(float f)
{
    return f - floorf(f);
}

static vec3_t ray_cast(__global struct scene *scene, const vec3_t *orig,
		       const vec3_t *dir, uint32_t depth)
{
	struct intersection isect;

	vec3_t hit_point, hit_normal, hit_color;
	vec2_t hit_tex_coords;
	bool hit;

	if (depth > scene->ray_depth)
		return scene->backcolor;

	hit = trace(scene, orig, dir, &isect, PRIMARY_RAY);
	if (!hit)
		return scene->backcolor;

	/* Evaluate surface properties (P, N, texture coordinates, etc.) */
	hit_point = v3_add(*orig, v3_muls(*dir, isect.near));
	object_get_surface_props(isect.hit_object, &hit_point, dir, isect.index,
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
	case MATERIAL_DIFFUSE: {
		/*
		 * Light loop (loop over all lights in the scene
		 * and accumulate their contribution)
		 */
		__global struct light *light;

		hit_color = vec3(0.0f, 0.0f, 0.0f);

		list_for_each_entry(light, &scene->lights, entry) {
			vec3_t light_dir, light_intensity;
			vec3_t point, rev_light_dir;
			vec3_t diffuse;

			struct intersection isect_shadow;
			float near;
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

			diffuse = v3_muls(light_intensity, isect.hit_object->albedo *
					  MAX(0.0f, v3_dot(hit_normal, rev_light_dir)));
			hit_color = v3_add(hit_color, diffuse);
		}
		break;
	}
	case MATERIAL_REFLECT: {
		vec3_t reflect_dir;

		reflect_dir = reflect(dir, &hit_normal);

		hit_point = v3_add(hit_point, v3_muls(hit_normal, scene->bias));
		hit_color = ray_cast(scene, &hit_point, &reflect_dir, depth + 1);
		/* Losing energy on reflection, pure average */
		hit_color = v3_muls(hit_color, 0.8f);
		break;
	}
	case MATERIAL_REFLECT_REFRACT: {
		vec3_t refract_color = vec3(0.0f, 0.0f, 0.0f);
		vec3_t reflect_color = vec3(0.0f, 0.0f, 0.0f);
		vec3_t reflect_orig, reflect_dir, bias;
		bool outside;
		float kr;

		kr = fresnel(dir, &hit_normal, isect.hit_object->ior);
		outside = v3_dot(*dir, hit_normal) < 0.0f;
		bias = v3_muls(hit_normal, scene->bias);


		/* compute refraction if it is not a case of total internal reflection */
		if (kr < 1.0f) {
			vec3_t refract_orig, refract_dir;

			refract_dir = refract(dir, &hit_normal, isect.hit_object->ior);
			refract_dir = v3_norm(refract_dir);

			refract_orig = outside ?
				v3_sub(hit_point, bias) :
				v3_add(hit_point, bias);

			refract_color = ray_cast(scene, &refract_orig, &refract_dir, depth + 1);
			refract_color = v3_muls(refract_color, 1 - kr);
		}
		reflect_dir = reflect(dir, &hit_normal);
		reflect_dir = v3_norm(reflect_dir);

		reflect_orig = outside ?
			v3_add(hit_point, bias) :
			v3_sub(hit_point, bias);

		reflect_color = ray_cast(scene, &reflect_orig, &reflect_dir, depth + 1);
		reflect_color = v3_muls(reflect_color, kr);

		hit_color = v3_add(reflect_color, refract_color);
		break;
	}
	default:
		hit_color = scene->backcolor;
		break;
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

	color = ray_cast(scene, &orig, &dir, 1);
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

static void triangle_mesh_destroy(struct object *obj)
{
	struct triangle_mesh *mesh =
		container_of(obj, struct triangle_mesh, obj);

	buf_destroy(mesh->vertices);
	buf_destroy(mesh->normals);
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

static int no_opencl;
static int one_frame;

enum {
	OPT_FOV = 'a',
	OPT_SCREEN_WIDTH,
	OPT_SCREEN_HEIGHT,

	OPT_CAM_PITCH,
	OPT_CAM_YAW,
	OPT_CAM_POS,

	OPT_BACKCOLOR,
	OPT_RAY_DEPTH,

	OPT_LIGHT,
	OPT_OBJECT,
};

static struct option long_options[] = {
	{"no-opencl", no_argument,	 &no_opencl, 1},
	{"opencl",    no_argument,	 &no_opencl, 0},
	{"one-frame", no_argument,	 &one_frame, 1},
	{"fov",	      required_argument, 0, OPT_FOV},
	{"width",     required_argument, 0, OPT_SCREEN_WIDTH},
	{"height",    required_argument, 0, OPT_SCREEN_HEIGHT},
	{"pitch",     required_argument, 0, OPT_CAM_PITCH},
	{"yaw",	      required_argument, 0, OPT_CAM_YAW},
	{"pos",	      required_argument, 0, OPT_CAM_POS},
	{"backcolor", required_argument, 0, OPT_BACKCOLOR},
	{"ray-depth", required_argument, 0, OPT_RAY_DEPTH},
	{"light",     required_argument, 0, OPT_LIGHT},
	{"object",    required_argument, 0, OPT_OBJECT},

	{0, 0, 0, 0}
};

enum {
	OBJECT_TYPE,
	OBJECT_MATERIAL,
	OBJECT_ROTATE_X,
	OBJECT_ROTATE_Y,
	OBJECT_ROTATE_Z,
	OBJECT_SCALE,
	OBJECT_TRANSLATE,
	OBJECT_ALBEDO,
	OBJECT_IOR,
	OBJECT_KD,
	OBJECT_KS,
	OBJECT_N,
	OBJECT_MESH_FILE,
	OBJECT_MESH_SMOOTH_SHADING,
	OBJECT_SPHERE_RADIUS,
	OBJECT_SPHERE_POS,
};

static char *const object_token[] = {
	[OBJECT_TYPE]	       = "type",
	[OBJECT_MATERIAL]      = "material",
	[OBJECT_ROTATE_X]      = "rotate-x",
	[OBJECT_ROTATE_Y]      = "rotate-y",
	[OBJECT_ROTATE_Z]      = "rotate-z",
	[OBJECT_SCALE]	       = "scale",
	[OBJECT_TRANSLATE]     = "translate",
	[OBJECT_ALBEDO]	       = "albedo",
	[OBJECT_IOR]	       = "ior",
	[OBJECT_KD]	       = "Kd",
	[OBJECT_KS]	       = "Ks",
	[OBJECT_N]	       = "n",
	[OBJECT_MESH_FILE]     = "file",
	[OBJECT_MESH_SMOOTH_SHADING] = "smooth-shading",
	[OBJECT_SPHERE_RADIUS] = "radius",
	[OBJECT_SPHERE_POS]    = "pos",
	NULL
};

enum object_type {
	UNKNOWN_OBJECT = 0,
	SPHERE_OBJECT,
	MESH_OBJECT,
};

struct object_params {
	int    parsed_params_bits;
	mat4_t o2w;
	enum object_type   type;
	enum material_type material;
	float  albedo;
	float  ior;
	float  Kd;
	float  Ks;
	float  n;
	struct {
		char  file[512];
		bool  smooth_shading;
	} mesh;
	struct {
		float  radius;
		vec3_t pos;
	} sphere;
};

static void default_object_params(struct object_params *params)
{
	memset(params, 0, sizeof(*params));
	params->material = MATERIAL_PHONG;
	params->albedo = 0.18f;
	params->ior = 1.3f;
	params->Kd = 0.8f;
	params->Ks = 0.2f;
	params->n = 10.0f;
	params->sphere.radius = 0.5f;
	params->sphere.pos = vec3(0.0f, 0.0f, 0.0f);
	params->o2w = m4_identity();
}

static void object_init(struct object *obj, struct object_ops *ops,
			struct object_params *params)
{
	INIT_LIST_HEAD(&obj->entry);
	obj->ops = *ops;
	obj->o2w = params->o2w;
	obj->material = params->material;
	obj->albedo = params->albedo;
	obj->ior = params->ior;
	obj->Kd = params->Kd;
	obj->Ks = params->Ks;
	obj->n = params->n;
}

static void sphere_set_radius(struct sphere *sphere, float radius)
{
	sphere->radius = radius;
	sphere->radius_pow2 = radius * radius;
}

static void sphere_set_pos(struct sphere *sphere, vec3_t pos)
{
	sphere->obj.o2w.m30 = pos.x;
	sphere->obj.o2w.m31 = pos.y;
	sphere->obj.o2w.m32 = pos.z;
	sphere->center = pos;
}

static void sphere_init(struct sphere *sphere, struct object_params *params)
{
	object_init(&sphere->obj, &sphere_ops, params);
	sphere_set_radius(sphere, params->sphere.radius);
	sphere_set_pos(sphere, params->sphere.pos);
}

static void triangle_mesh_init(struct opencl *opencl, struct object_params *params,
			       struct triangle_mesh *mesh, uint32_t num_verts,
			       vec3_t *verts, vec3_t *normals, vec2_t *sts)
{
	vec3_t *P, *N;
	vec2_t *S;
	mat4_t transform_normals;
	int i;

	P = buf_allocate(opencl, num_verts * sizeof(*P));
	assert(P);

	N = buf_allocate(opencl, num_verts * sizeof(*N));
	assert(N);

	S = buf_allocate(opencl, num_verts * sizeof(*S));
	assert(S);

	/*
	 * Computing the transpose of the object-to-world inverse matrix.
	 * We can't transform normal by multiplying it on o2w matrix as we
	 * did for each vertex, because normal will cease to be perpendicular,
	 * so 'N dot V == 0' will not be true any more. What we need instead
	 * is to keep same rotation, but invert scaling, i.e. if we have
	 * a transformation matrix: M = R * S, which implies scaling and
	 * rotation, we need keep rotation but invert scaling, thus:
	 *   M' = R * S(-1)
	 * The inverse of a rotation matrix is its transpose, the transpose of
	 * a scale matrix is the same scale matrix (so noop), thus to get the
	 * transformation matrix for normal we can:
	 *
	 *  M' = M(-1)(T) = (R * S)(-1)(T) = R(-1)(T) * S(-1)(T) = R * S(-1)
	 *
	 * Corresponding math:
	 *   v . n = 0
	 *   v . M*M(-1) . n = 0        [because I = M*M(-1)]
	 *   (v*M) . (n*M(-1)(T)) = 0   [because A*x . y = x . A(T)*y]
	 *
	 * where v - vertex, n - normal, M - transformation matrix
	 */
	transform_normals = m4_transpose(m4_invert_affine(params->o2w));

	/* Expect triangulated mesh */
	assert(!(num_verts % 3));

	/* For each triangle */
	for (i = 0; i < num_verts; i += 3) {
		/* Transform vertices */
		P[i + 0] = m4_mul_pos(params->o2w, verts[i + 0]);
		P[i + 1] = m4_mul_pos(params->o2w, verts[i + 1]);
		P[i + 2] = m4_mul_pos(params->o2w, verts[i + 2]);

		/* Transform normals */
		N[i + 0] = m4_mul_dir(transform_normals, normals[i + 0]);
		N[i + 1] = m4_mul_dir(transform_normals, normals[i + 1]);
		N[i + 2] = m4_mul_dir(transform_normals, normals[i + 2]);

		N[i + 0] = v3_norm(N[i + 0]);
		N[i + 1] = v3_norm(N[i + 1]);
		N[i + 2] = v3_norm(N[i + 2]);

		S[i + 0] = sts[i + 0];
		S[i + 1] = sts[i + 1];
		S[i + 2] = sts[i + 2];
	}

	/* Init object */
	object_init(&mesh->obj, &triangle_mesh_ops, params);
	mesh->smooth_shading = params->mesh.smooth_shading;
	mesh->num_verts = num_verts;
	mesh->vertices = P;
	mesh->normals = N;
	mesh->sts = S;

	/* Not supposed to changed by the host, so unmap immediately */
	buf_unmap(P);
	buf_unmap(N);
	buf_unmap(S);
}

static void
triangle_mesh_init_geo(struct opencl *opencl, struct object_params *params,
		       struct triangle_mesh *mesh, uint32_t nfaces,
		       uint32_t *face_index, uint32_t *verts_index,
		       vec3_t *verts, vec3_t *normals, vec2_t *sts)
{
	uint32_t i, j, l, k = 0, max_vert_index = 0;
	uint32_t num_tris = 0, num_verts;
	vec3_t *flat_verts, *flat_norms;
	vec2_t *flat_sts;

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
	num_verts = num_tris * 3;
	assert(max_vert_index <= num_verts);

	flat_verts = calloc(num_verts, sizeof(*flat_verts));
	assert(flat_verts);
	flat_norms = calloc(num_verts, sizeof(*flat_norms));
	assert(flat_norms);
	flat_sts = calloc(num_verts, sizeof(*flat_sts));
	assert(flat_sts);

	/* For each face */
	for (i = 0, k = 0, l = 0; i < nfaces; i++) {
		/* For each triangle in a face */
		for (j = 0; j < face_index[i] - 2; j++, l += 3) {
			assert(l + 2 < num_verts);

			/* Flatten vertices */
			flat_verts[l + 0] = verts[verts_index[k]];
			flat_verts[l + 1] = verts[verts_index[k + j + 1]];
			flat_verts[l + 2] = verts[verts_index[k + j + 2]];

			/* Flatten normals */
			flat_norms[l + 0] = normals[k];
			flat_norms[l + 1] = normals[k + j + 1];
			flat_norms[l + 2] = normals[k + j + 2];

			/* Flatten texture coords */
			flat_sts[l + 0] = sts[k];
			flat_sts[l + 1] = sts[k + j + 1];
			flat_sts[l + 2] = sts[k + j + 2];
		}
		k += face_index[i];
	}
	triangle_mesh_init(opencl, params, mesh, num_verts, flat_verts,
			   flat_norms, flat_sts);
	free(flat_verts);
	free(flat_norms);
	free(flat_sts);
}

static int triangle_mesh_load_geo(struct scene *scene,
				  struct object_params *params)
{
	uint32_t num_faces, verts_ind_arr_sz, verts_arr_sz;
	int ret, i;
	size_t pos;
	FILE *f;

	uint32_t *face_index, *verts_index;
	vec3_t *verts, *normals;
	vec2_t *sts;

	struct triangle_mesh *mesh;

	f = fopen(params->mesh.file, "r");
	if (!f) {
		fprintf(stderr, "Can't open file: %s\n", params->mesh.file);
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
		ret = fscanf(f, "%f %f %f ", &vert->x, &vert->y, &vert->z);
		assert(ret == 3);
	}

	normals = calloc(verts_ind_arr_sz, sizeof(*normals));
	assert(normals);

	for (i = 0; i < verts_ind_arr_sz; i++) {
		vec3_t *norm = &normals[i];
		ret = fscanf(f, "%f %f %f ", &norm->x, &norm->y, &norm->z);
		assert(ret == 3);
	}

	sts = calloc(verts_ind_arr_sz, sizeof(*sts));
	assert(sts);

	for (i = 0; i < verts_ind_arr_sz; i++) {
		vec2_t *coord = &sts[i];
		ret = fscanf(f, "%f %f ", &coord->x, &coord->y);
		assert(ret == 2);
	}

	pos = ftell(f);
	fseek(f, 0, SEEK_END);
	/* The whole file was parsed */
	assert(pos == ftell(f));
	fclose(f);

	mesh = buf_allocate(scene->opencl, sizeof(*mesh));
	if (!mesh) {
		ret = -ENOMEM;
		goto error;
	}
	triangle_mesh_init_geo(scene->opencl, params, mesh, num_faces,
			       face_index, verts_index, verts, normals, sts);
	list_add_tail(&mesh->obj.entry, &scene->objects);
	ret = 0;

error:
	free(face_index);
	free(verts_index);
	free(verts);
	free(normals);
	free(sts);

	return ret;
}

static int triangle_mesh_load_obj(struct scene *scene,
				  struct object_params *params)
{
	const struct aiScene *ai_scene;
	struct object *obj, *tmp;
	LIST_HEAD(objects);

	vec3_t *flat_verts, *flat_norms;
	vec2_t *flat_sts;

	uint32_t num_verts, i_mesh;
	int ret, i;

	ai_scene = aiImportFile(params->mesh.file,
				aiProcess_CalcTangentSpace	 |
				aiProcess_Triangulate		 |
				aiProcess_JoinIdenticalVertices	 |
				aiProcess_SortByPType		 |
				(params->mesh.smooth_shading ?
				 aiProcess_GenSmoothNormals :
				 aiProcess_GenNormals));
	if (!ai_scene) {
		printf("Can't open %s, aiImportFile failed\n", params->mesh.file);
		return -EINVAL;
	}

	/* Count all vertices */
	for (num_verts = 0, i_mesh = 0; i_mesh < ai_scene->mNumMeshes; i_mesh++) {
		const struct aiMesh *ai_mesh;
		uint32_t i_face;

		ai_mesh = ai_scene->mMeshes[i_mesh];
		for (i_face = 0; i_face < ai_mesh->mNumFaces; i_face++) {
			const struct aiFace *ai_face = &ai_mesh->mFaces[i_face];

			assert(ai_face->mNumIndices == 3);
			num_verts += ai_face->mNumIndices;
		}
	}

	flat_verts = calloc(num_verts, sizeof(*flat_verts));
	assert(flat_verts);
	flat_norms = calloc(num_verts, sizeof(*flat_norms));
	assert(flat_norms);
	flat_sts = calloc(num_verts, sizeof(*flat_sts));
	assert(flat_sts);

	/* Flatten vertices, normals and texture coords */
	for (i = 0, i_mesh = 0; i_mesh < ai_scene->mNumMeshes; i_mesh++) {
		const struct aiMesh *ai_mesh;
		struct triangle_mesh *mesh;
		uint32_t i_face;

		ai_mesh = ai_scene->mMeshes[i_mesh];
		for (i_face = 0; i_face < ai_mesh->mNumFaces; i_face++) {
			const struct aiFace *ai_face = &ai_mesh->mFaces[i_face];
			uint32_t i_ind;

			for (i_ind = 0; i_ind < ai_face->mNumIndices / 3; i_ind++, i += 3) {
				struct aiVector3D *v0, *v1, *v2;

				/* Flatten vertices */
				v0 = &ai_mesh->mVertices[ai_face->mIndices[i_ind * 3 + 0]];
				v1 = &ai_mesh->mVertices[ai_face->mIndices[i_ind * 3 + 1]];
				v2 = &ai_mesh->mVertices[ai_face->mIndices[i_ind * 3 + 2]];

				flat_verts[i + 0] = vec3(v0->x, v0->y, v0->z);
				flat_verts[i + 1] = vec3(v1->x, v1->y, v1->z);
				flat_verts[i + 2] = vec3(v2->x, v2->y, v2->z);

				/* Flatten normals */
				v0 = &ai_mesh->mNormals[ai_face->mIndices[i_ind * 3 + 0]];
				v1 = &ai_mesh->mNormals[ai_face->mIndices[i_ind * 3 + 1]];
				v2 = &ai_mesh->mNormals[ai_face->mIndices[i_ind * 3 + 2]];

				flat_norms[i + 0] = vec3(v0->x, v0->y, v0->z);
				flat_norms[i + 1] = vec3(v1->x, v1->y, v1->z);
				flat_norms[i + 2] = vec3(v2->x, v2->y, v2->z);

				/* Flatten texture coords */
				v0 = &ai_mesh->mTextureCoords[0][ai_face->mIndices[i_ind * 3 + 0]];
				v1 = &ai_mesh->mTextureCoords[0][ai_face->mIndices[i_ind * 3 + 1]];
				v2 = &ai_mesh->mTextureCoords[0][ai_face->mIndices[i_ind * 3 + 2]];

				flat_sts[i + 0] = vec2(v0->x, v0->y);
				flat_sts[i + 1] = vec2(v1->x, v1->y);
				flat_sts[i + 2] = vec2(v2->x, v2->y);
			}
		}

		mesh = buf_allocate(scene->opencl, sizeof(*mesh));
		if (!mesh) {
			ret = -ENOMEM;
			goto error;
		}
		triangle_mesh_init(scene->opencl, params, mesh, num_verts,
				   flat_verts, flat_norms, flat_sts);
		list_add_tail(&mesh->obj.entry, &objects);

	}
	list_splice_tail(&objects, &scene->objects);
	ret = 0;
out:
	aiReleaseImport(ai_scene);
	free(flat_verts);
	free(flat_norms);
	free(flat_sts);

	return ret;

error:
	list_for_each_entry_safe(obj, tmp, &objects, entry)
		object_destroy(obj);

	goto out;
}

static bool is_parsed_object_param(struct object_params *params, int t)
{
	return params->parsed_params_bits & (1<<t);
}

static int parse_object_params(char *subopts, struct object_params *params)
{
	int errfnd = 0, num, ret;
	char *value;

	default_object_params(params);

	while (*subopts != '\0' && !errfnd) {
		char *real_value;
		float *fptr = NULL;

		int c = getsubopt(&subopts, object_token, &value);

		/*
		 * Return comma to the string in order to parse several times,
		 * but keep real value as dupa.
		 */
		real_value = strdupa(value);
		if (c != -1 && *subopts)
			*(subopts - 1) = ',';

		switch (c) {
		case OBJECT_TYPE:
			if (!strcmp(real_value, "sphere"))
				params->type = SPHERE_OBJECT;
			else if (!strcmp(real_value, "mesh"))
				params->type = MESH_OBJECT;
			else {
				fprintf(stderr, "Invalid object type '%s'\n",
					real_value);
				return -EINVAL;
			}
			break;
		case OBJECT_MATERIAL:
			if (!strcmp(real_value, "phong"))
				params->material = MATERIAL_PHONG;
			else if (!strcmp(real_value, "diffuse"))
				params->material = MATERIAL_DIFFUSE;
			else if (!strcmp(real_value, "reflect"))
				params->material = MATERIAL_REFLECT;
			else if (!strcmp(real_value, "reflect-refract"))
				params->material = MATERIAL_REFLECT_REFRACT;
			else {
				fprintf(stderr, "Unknown material specified\n");
				return -EINVAL;
			}
			break;
		case OBJECT_ROTATE_X:
		case OBJECT_ROTATE_Y:
		case OBJECT_ROTATE_Z: {
			float deg_angle;
			mat4_t m;

			ret = sscanf(value, "%f", &deg_angle);
			if (ret != 1) {
				fprintf(stderr, "Invalid object '%s' parameter\n",
					object_token[c]);
				return -EINVAL;
			}
			if (c == OBJECT_ROTATE_X)
				m = m4_rotation_x(deg2rad(deg_angle));
			else if (c == OBJECT_ROTATE_Y)
				m = m4_rotation_y(deg2rad(deg_angle));
			else
				m = m4_rotation_z(deg2rad(deg_angle));
			params->o2w = m4_mul(params->o2w, m);
			break;
		}
		case OBJECT_SCALE:
		case OBJECT_TRANSLATE: {
			vec3_t vec;
			mat4_t m;

			ret = sscanf(value, "%f,%f,%f%n", &vec.x, &vec.y, &vec.z,
				     &num);
			if (ret != 3) {
				ret = sscanf(value, "%f%n", &vec.x, &num);
				if (ret != 1) {
					fprintf(stderr, "Invalid object '%s' parameter\n",
						object_token[c]);
					return -EINVAL;
				}
				/* Apply single value to all others */
				vec.y = vec.z = vec.x;
			}
			subopts = value + num;
			if (subopts[0] == ',')
				/* Skip trailing comma */
				subopts += 1;
			if (c == OBJECT_SCALE)
				m = m4_scaling(vec);
			else
				m = m4_translation(vec);
			params->o2w = m4_mul(params->o2w, m);
			break;
		}
		case OBJECT_ALBEDO:
			fptr = &params->albedo;
			break;
		case OBJECT_IOR:
			fptr = &params->ior;
			break;
		case OBJECT_KD:
			fptr = &params->Kd;
			break;
		case OBJECT_KS:
			fptr = &params->Ks;
			break;
		case OBJECT_N:
			fptr = &params->n;
			break;
		case OBJECT_SPHERE_RADIUS: {
			fptr = &params->sphere.radius;
			break;
		}
		case OBJECT_SPHERE_POS: {
			ret = sscanf(value, "%f,%f,%f%n", &params->sphere.pos.x,
				     &params->sphere.pos.y, &params->sphere.pos.z,
				     &num);
			if (ret != 3) {
				fprintf(stderr, "Invalid object '%s' parameter\n",
					object_token[c]);
				return -EINVAL;
			}
			subopts = value + num;
			if (subopts[0] == ',')
				/* Skip trailing comma */
				subopts += 1;
			break;
		}
		case OBJECT_MESH_FILE: {
			char *file;

			ret = sscanf(value, "%m[^,]", &file);
			if (ret != 1) {
				fprintf(stderr, "Invald object '%s' parameter\n",
					object_token[c]);
				return -EINVAL;
			}
			ret = snprintf(params->mesh.file, sizeof(params->mesh.file),
				       "%s", file);
			free(file);
			if (ret >= sizeof(params->mesh.file)) {
				fprintf(stderr, "Object '%s' parameter is too big\n",
					object_token[c]);
				return -EINVAL;
			}
			break;
		}
		case OBJECT_MESH_SMOOTH_SHADING: {
			char *flag;

			ret = sscanf(value, "%m[^,]", &flag);
			if (ret != 1) {
				fprintf(stderr, "Invald object '%s' parameter\n",
					object_token[c]);
				return -EINVAL;
			}
			if (!strcmp(flag, "0") || !strcmp(flag, "false"))
				params->mesh.smooth_shading = false;
			else if (!strcmp(flag, "1") || !strcmp(flag, "true"))
				params->mesh.smooth_shading = true;
			else {
				fprintf(stderr, "Invalid value of  '%s' parameter, should be '1','0','true or 'false'\n",
					object_token[c]);
				free(flag);
				return -EINVAL;
			}
			free(flag);
			break;
		}
		default:
			fprintf(stderr, "Unknown object parameter: %s\n",
				value);
			return -EINVAL;
		}
		/* Common param */
		if (fptr) {
			ret = sscanf(value, "%f", fptr);
			if (ret != 1) {
				fprintf(stderr, "Invald object '%s' parameter\n",
					object_token[c]);
				return -EINVAL;
			}
		}
		params->parsed_params_bits |= (1<<c);
	}

	/* Validate parameters */
	if (!is_parsed_object_param(params, OBJECT_TYPE)) {
		fprintf(stderr, "Object type is not specified\n");
		return -EINVAL;
	}
	switch (params->type) {
	case SPHERE_OBJECT:
		if (is_parsed_object_param(params, OBJECT_MESH_FILE)) {
			fprintf(stderr, "Invalid parameter '%s' for 'sphere' object type\n",
				object_token[OBJECT_MESH_FILE]);
			return -EINVAL;
		}
		break;
	case MESH_OBJECT:
		if (!is_parsed_object_param(params, OBJECT_MESH_FILE)) {
			fprintf(stderr, "Required parameter 'file' for 'mesh' object is not specified\n");
			return -EINVAL;
		}
		if (is_parsed_object_param(params, OBJECT_SPHERE_RADIUS)) {
			fprintf(stderr, "Invalid parameter '%s' for 'mesh' object type\n",
				object_token[OBJECT_SPHERE_RADIUS]);
			return -EINVAL;
		}
		if (is_parsed_object_param(params, OBJECT_SPHERE_POS)) {
			fprintf(stderr, "Invalid parameter '%s' for 'mesh' object type\n",
				object_token[OBJECT_SPHERE_POS]);
			return -EINVAL;
		}
		break;
	default:
		fprintf(stderr, "Unknown object type\n");
		return -EINVAL;
	}

	return 0;
}

static void objects_destroy(struct scene *scene)
{
	struct object *obj, *tmp;

	list_for_each_entry_safe(obj, tmp, &scene->objects, entry)
		object_destroy(obj);
}

static int objects_create_from_params(struct scene *scene,
				      struct object_params *params)
{
	switch (params->type) {
	case SPHERE_OBJECT: {
		struct sphere *sphere;

		sphere = buf_allocate(scene->opencl, sizeof(*sphere));
		if (!sphere)
			return -ENOMEM;
		sphere_init(sphere, params);
		list_add_tail(&sphere->obj.entry, &scene->objects);
		return 0;
	}
	case MESH_OBJECT: {
		int len = strlen(params->mesh.file);

		if (len > 3 &&
		    !strcmp(params->mesh.file + len - 4, ".geo")) {
			return triangle_mesh_load_geo(scene, params);
		} else if (len > 3 &&
			   !strcmp(params->mesh.file + len - 4, ".obj")) {
			return triangle_mesh_load_obj(scene, params);
		}
		fprintf(stderr, "Invalid object file extension\n");
		return -EINVAL;
	}
	default:
		/* Params already validated */
		assert(0);
		return -EINVAL;
	}
}

static int objects_create(struct scene *scene, int argc, char **argv)
{
	int ret;

	optind = 1;
	while (1) {
		int c, option_index = 0;

		c = getopt_long(argc, argv, "", long_options, &option_index);
		if (c == -1)
			break;

		/* Create object */
		switch (c) {
		case OPT_OBJECT: {
			struct object_params params;

			ret = parse_object_params(optarg, &params);
			if (ret)
				goto error;
			ret = objects_create_from_params(scene, &params);
			if (ret)
				goto error;
			break;
		}
		default:
			break;
		}
	}

	return 0;

error:
	objects_destroy(scene);
	return ret;
}

static void light_init(struct light *light, struct light_ops *ops,
		       const vec3_t *color, float intensity)
{
	INIT_LIST_HEAD(&light->entry);
	light->ops = *ops;
	light->color = *color;
	light->intensity = intensity;
}

static void distant_light_destroy(struct light *light)
{
	struct distant_light *dlight =
		container_of(light, struct distant_light, light);

	buf_destroy(dlight);
}

static int distant_light_unmap(struct light *light)
{
	struct distant_light *dlight =
		container_of(light, struct distant_light, light);

	return buf_unmap(dlight);
}

struct light_ops distant_light_ops = {
	.destroy	 = distant_light_destroy,
	.unmap		 = distant_light_unmap,
	.illuminate	 = distant_light_illuminate,
	.illuminate_type = DISTANT_LIGHT_ILLUMINATE,
};

static void distant_light_set_dir(struct distant_light *dlight, vec3_t dir)
{
	dlight->dir = v3_norm(dir);
}

static void distant_light_init(struct distant_light *dlight, const vec3_t *color,
			       float intensity)
{
	light_init(&dlight->light, &distant_light_ops, color, intensity);
	distant_light_set_dir(dlight, vec3(0.0f, 0.0f, -1.0f));
}

static void point_light_destroy(struct light *light)
{
	struct point_light *plight =
		container_of(light, struct point_light, light);

	buf_destroy(plight);
}

static int point_light_unmap(struct light *light)
{
	struct point_light *plight =
		container_of(light, struct point_light, light);

	return buf_unmap(plight);
}

struct light_ops point_light_ops = {
	.destroy	 = point_light_destroy,
	.unmap		 = point_light_unmap,
	.illuminate	 = point_light_illuminate,
	.illuminate_type = POINT_LIGHT_ILLUMINATE,
};

static void point_light_init(struct point_light *plight, const vec3_t *color,
			     float intensity)
{
	light_init(&plight->light, &point_light_ops, color, intensity);
	plight->pos = vec3(0.0f, 1.0f, 0.0f);
}

enum {
	LIGHT_TYPE,
	LIGHT_COLOR,
	LIGHT_INTENSITY,
	LIGHT_DIR,
	LIGHT_POS,
};

static char *const light_token[] = {
	[LIGHT_TYPE]	  = "type",
	[LIGHT_COLOR]	  = "color",
	[LIGHT_INTENSITY] = "intensity",
	[LIGHT_DIR]	  = "dir",
	[LIGHT_POS]	  = "pos",
};

enum {
	UNKNOWN_LIGHT = 0,
	DISTANT_LIGHT,
	POINT_LIGHT,
};

static int parse_light_type_param(char *subopts)
{
	int errfnd = 0, type;
	char *value;

	type = UNKNOWN_LIGHT;
	while (*subopts != '\0' && !errfnd) {
		int c = getsubopt(&subopts, light_token, &value);

		switch (c) {
		case LIGHT_TYPE:
			if (!strcmp(value, "distant"))
				type = DISTANT_LIGHT;
			else if (!strcmp(value, "point"))
				type = POINT_LIGHT;
			else {
				type = -EINVAL;
				fprintf(stderr, "Invalid light type '%s'\n",
					value);
			}
			break;
		default:
			break;
		}

		/* Don't modify opts string in order to parse several times */
		if (*subopts)
			*(subopts - 1) = ',';

		if (type != UNKNOWN_LIGHT)
			break;
	}

	return type;
}

static int parse_light_params(char *subopts, int light_type, struct light *light)
{
	int errfnd = 0, ret, num;
	char *value;

	while (*subopts != '\0' && !errfnd) {
		int c = getsubopt(&subopts, light_token, &value);

		/* Don't modify opts string in order to parse several times */
		if (c != -1 && *subopts)
			*(subopts - 1) = ',';

		switch (c) {
		case LIGHT_TYPE:
			/* See parse_light_type_param() */
			break;
		case LIGHT_COLOR: {
			uint32_t color;
			ret = sscanf(value, "%x", &color);
			if (ret != 1) {
				fprintf(stderr, "Invalid light color, should be hex.\n");
				return -EINVAL;
			}
			light->color.x = ((color>>16) & 0xff) / 255.0f;
			light->color.y = ((color>>8) & 0xff) / 255.0f;
			light->color.z = (color & 0xff) / 255.0f;
			break;
		}
		case LIGHT_INTENSITY:
			ret = sscanf(value, "%f", &light->intensity);
			if (ret != 1) {
				fprintf(stderr, "Invalid light intensity, should be float.\n");
				return -EINVAL;
			}
			break;
		case LIGHT_DIR: {
			struct distant_light *dlight;

			if (light_type != DISTANT_LIGHT) {
				fprintf(stderr, "Invalid parameter '%s' for this type of light.\n",
					light_token[c]);
				return -EINVAL;
			}
			dlight = container_of(light, typeof(*dlight), light);
			ret = sscanf(value, "%f,%f,%f%n", &dlight->dir.x,
				     &dlight->dir.y, &dlight->dir.z, &num);
			if (ret != 3) {
				fprintf(stderr, "Invalid distant light direction, should be float,float,float.\n");
				return -EINVAL;
			}
			distant_light_set_dir(dlight, dlight->dir);
			subopts = value + num;
			if (subopts[0] == ',')
				/* Skip trailing comma */
				subopts += 1;
			break;
		}
		case LIGHT_POS: {
			struct point_light *plight;

			if (light_type != POINT_LIGHT) {
				fprintf(stderr, "Invalid parameter '%s' for this type of light.\n",
					light_token[c]);
				return -EINVAL;
			}
			plight = container_of(light, typeof(*plight), light);
			ret = sscanf(value, "%f,%f,%f%n", &plight->pos.x,
				     &plight->pos.y, &plight->pos.z, &num);
			if (ret != 3) {
				fprintf(stderr, "Invalid point light position, should be float,float,float.\n");
				return -EINVAL;
			}
			subopts = value + num;
			if (subopts[0] == ',')
				/* Skip trailing comma */
				subopts += 1;
			break;
		}
		default:
			fprintf(stderr, "Unknown light parameter: %s\n",
				value);
			return -EINVAL;
		}
	}

	return 0;
}

static void light_destroy(struct light *light)
{
	list_del(&light->entry);
	light->ops.destroy(light);
}

static void lights_destroy(struct scene *scene)
{
	struct light *light, *tmp;

	list_for_each_entry_safe(light, tmp, &scene->lights, entry)
		light_destroy(light);
}

static struct light *light_create(struct opencl *opencl, int light_type)
{
	vec3_t color = vec3(1.0f, 1.0f, 1.0f);
	float intensity = 5.0f;

	switch (light_type) {
	case DISTANT_LIGHT: {
		struct distant_light *dlight;

		dlight =  buf_allocate(opencl, sizeof(*dlight));
		if (!dlight)
			return NULL;
		distant_light_init(dlight, &color, intensity);
		return &dlight->light;
	}
	case POINT_LIGHT: {
		struct point_light *plight;

		plight =  buf_allocate(opencl, sizeof(*plight));
		if (!plight)
			return NULL;
		point_light_init(plight, &color, intensity);
		return &plight->light;
	}
	default:
		assert(0);
		return NULL;
	}
}

static int lights_create(struct scene *scene, int argc, char **argv)
{
	int ret = 0;

	optind = 1;
	while (1) {
		int c, option_index = 0;
		int light_type;

		struct light *light;

		c = getopt_long(argc, argv, "", long_options, &option_index);
		if (c == -1)
			break;

		/* Create light */
		switch (c) {
		case OPT_LIGHT:
			light_type = parse_light_type_param(optarg);
			if (light_type < 0) {
				ret = light_type;
				goto error;
			}
			if (light_type == UNKNOWN_LIGHT) {
				fprintf(stderr, "Light type is not specified\n");
				ret = -EINVAL;
				goto error;
			}
			light = light_create(scene->opencl, light_type);
			if (!light) {
				ret = -ENOMEM;
				goto error;
			}
			ret = parse_light_params(optarg, light_type, light);
			if (ret) {
				light_destroy(light);
				goto error;
			}
			list_add_tail(&light->entry, &scene->lights);
			break;
		default:
			break;
		}
	}

	return 0;

error:
	lights_destroy(scene);
	return ret;
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
	ret = TTF_Init();
	if (ret) {
		SDL_Quit();
		printf("Can't init TTF\n");
		return -1;
	}

	sdl = malloc(sizeof(*sdl));
	assert(sdl);

	sdl->window = SDL_CreateWindow("YART", SDL_WINDOWPOS_CENTERED,
				       SDL_WINDOWPOS_CENTERED,
				       scene->width, scene->height,
				       SDL_WINDOW_HIDDEN);
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
	TTF_Quit();
	SDL_Quit();
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
				  float cam_yaw, float fov,
				  vec3_t backcolor, uint32_t ray_depth)
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
		.backcolor    = backcolor,
		.ray_depth    = ray_depth,
		.c2w	      = m4_identity(),
		.bias	      = 0.0001,
		.opencl	      = opencl,
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

			color = ray_cast(scene, &orig, &dir, 1);
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

/**
 * Welford's online algorithm
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 */
struct welford_state {
	uint64_t count;
	float mean, m2;
};

static float avg_welford(struct welford_state *s, float new_value)
{
	float delta, delta2;

	s->count += 1;
	delta = new_value - s->mean;
	s->mean += delta / s->count;
	delta2 = new_value - s->mean;
	s->m2 += delta * delta2;

	return s->mean;
}

static float scene_average_fps(struct scene *scene)
{
	static struct welford_state s;
	static uint64_t render_ns;
	float fps = 0;

	if (render_ns) {
		fps = 1000000000.0f / (nsecs() - render_ns);
		fps = avg_welford(&s, fps);
	}
	render_ns = nsecs();

	return fps;
}

static void draw_scene_status(struct scene *scene)
{
	SDL_Renderer *renderer = scene->sdl->renderer;
	SDL_Surface *rect_surface, *text_surface;
	SDL_Texture *text;
	TTF_Font *font;
	SDL_Rect r, rr;

	SDL_Color color = { 0xaa, 0xaa, 0xaa};

	char buf[512];

	r.x = scene->width - 120;
	r.y = 0;
	r.w = 120;
	r.h = 120;

	rect_surface = SDL_CreateRGBSurfaceWithFormat(0, 300, 400, 32,
						      SDL_PIXELFORMAT_RGBA8888);
	assert(rect_surface);

	SDL_SetRenderDrawColor(scene->sdl->renderer, 0x60, 0x60, 0x60, 0x90);
	SDL_RenderFillRect(renderer, &r);

	font = TTF_OpenFont("fonts/FreeMono.ttf", 14);
	assert(font);

	snprintf(buf, sizeof(buf), "    X %8.3f", scene->cam.pos.x);
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 0, text_surface->w, text_surface->h};
	SDL_BlitSurface(text_surface, NULL, rect_surface, &rr);
	SDL_FreeSurface(text_surface);

	snprintf(buf, sizeof(buf), "    Y %8.3f", scene->cam.pos.y);
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 15, text_surface->w, text_surface->h};
	SDL_BlitSurface(text_surface, NULL, rect_surface, &rr);
	SDL_FreeSurface(text_surface);

	snprintf(buf, sizeof(buf), "    Z %8.3f", scene->cam.pos.z);
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 30, text_surface->w, text_surface->h};
	SDL_BlitSurface(text_surface, NULL, rect_surface, &rr);
	SDL_FreeSurface(text_surface);

	snprintf(buf, sizeof(buf), "Pitch %8.3f", scene->cam.pitch);
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 50, text_surface->w, text_surface->h};
	SDL_BlitSurface(text_surface, NULL, rect_surface, &rr);
	SDL_FreeSurface(text_surface);

	snprintf(buf, sizeof(buf), "  Yaw %8.3f", scene->cam.yaw);
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 65, text_surface->w, text_surface->h};
	SDL_BlitSurface(text_surface, NULL, rect_surface, &rr);
	SDL_FreeSurface(text_surface);

	snprintf(buf, sizeof(buf), "  FPS %8.0f", scene_average_fps(scene));
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 85, text_surface->w, text_surface->h};
	SDL_BlitSurface(text_surface, NULL, rect_surface, &rr);
	SDL_FreeSurface(text_surface);

	text = SDL_CreateTextureFromSurface(renderer, rect_surface);
	assert(text);
	r = (SDL_Rect){ scene->width - 115, 10, rect_surface->w, rect_surface->h };
	SDL_FreeSurface(rect_surface);
	SDL_RenderCopy(renderer, text, NULL, &r);
	SDL_DestroyTexture(text);

	TTF_CloseFont(font);
}

static void render(struct scene *scene)
{
	struct sdl *sdl = scene->sdl;
	int ret;

	SDL_SetRelativeMouseMode(SDL_TRUE);
	SDL_StopTextInput();
	SDL_ShowWindow(sdl->window);

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
			camera_inc_angles(scene, -mouse.y * MOVE_SPEED,
					  mouse.x * MOVE_SPEED);
			updated_cam = true;
		}

		/* Handle keyboard */
		if (keyb[SDL_SCANCODE_W]) {
			cam->pos = v3_add(cam->pos, v3_muls(cam->dir, MOVE_SPEED));
			updated_cam = true;
		}
		else if (keyb[SDL_SCANCODE_S]) {
			cam->pos = v3_sub(cam->pos, v3_muls(cam->dir, MOVE_SPEED));
			updated_cam = true;
		}
		if (keyb[SDL_SCANCODE_A]) {
			vec3_t up = vec3(0.0f, 1.0f, 0.0f);
			vec3_t right = v3_cross(cam->dir, up);

			cam->pos = v3_sub(cam->pos, v3_muls(right, MOVE_SPEED));
			updated_cam = true;
		}
		else if (keyb[SDL_SCANCODE_D]) {
			vec3_t up = vec3(0.0f, 1.0f, 0.0f);
			vec3_t right = v3_cross(cam->dir, up);

			cam->pos = v3_add(cam->pos, v3_muls(right, MOVE_SPEED));
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

		SDL_RenderClear(sdl->renderer);

		/* Map for reading */
		ret = buf_map(scene->framebuffer, BUF_MAP_READ);
		assert(!ret);

		SDL_UpdateTexture(sdl->screen, NULL, scene->framebuffer,
				  scene->width * sizeof(*scene->framebuffer));

		/* Unmap */
		ret = buf_unmap(scene->framebuffer);
		assert(!ret);

		SDL_RenderCopy(sdl->renderer, sdl->screen, NULL, NULL);
		draw_scene_status(scene);
		SDL_RenderPresent(sdl->renderer);
	}
}

static void usage(void)
{
	printf("Usage:\n"
	       "  $ yart [--no-opencl] [--one-frame] [--fov <fov>] [--width <width>] [--height <height>]\n"
	       "         [--pitch <pitch>] [--yaw <yaw>] [--pos <pos>] [--light <light params>]... [--object <object params> ]..."
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
	       "                 e.g.: '--pos 0.0,1.0,12.0'\n"
	       "   --backcolor - background color in hex, e.g. for red ff0000\n"
	       "   --ray-depth - number of ray casting depth, 5 is default\n"
	       "\n"
	       "   --light     - add light, comma separated parameters should follow:\n"
	       "                 'type'      - required parameter, specifies type of the light, 'distant' or 'point'\n"
	       "                               can be specified\n"
	       "                 'color'     - RGB color in hex, e.g. for red ff0000\n"
	       "                 'intensity' - light intensity, should be float\n"
	       "                Distant light:\n"
	       "                 'dir'       - direction vector of light in infinity\n"
	       "                Point light:\n"
	       "                 'pos'        - position of the point light\n"
	       "\n"
	       "   --object    - add object, comma separated parameters should follow:\n"
	       "                 'type'      - required parameter, specifies type of the object, 'mesh' or 'sphere'\n"
	       "                               can be specified\n"
	       "                 'material'  - object material (shading), should be 'phong', 'diffuse', 'reflect', 'reflect-refract'\n"
	       "                 'rotate-x'\n"
	       "                 'rotate-y'\n"
	       "                 'rotate-z'  - rotate around axis by a give angle in degrees\n"
	       "                 'scale'     - scale on specified vector, accepts a single float or float,float,float\n"
	       "                 'translate' - translates on specified offset vector, accepts float,float,float\n"
	       "                 'albedo' - albedo\n"
	       "                 'ior'    - index of refraction\n"
	       "                 'Kd'     - diffuse weight\n"
	       "                 'Ks'     - specular weight\n"
	       "                 'n'      - specular exponent\n"
	       "                Sphere:\n"
	       "                 'radius' - sphere radius\n"
	       "                 'pos'    - spehere position\n"
	       "                Mesh:\n"
	       "                 'file'   - required paremeter, file path of the mesh object\n"
	       "                 e.g.: '--object type=sphere,radius=1.0,Ks=2.0,pos=1.0,0.1,0.3,n=5.0'\n"
	       "                 'smooth-shading' - enables smooth shading, should '0','1','false' or true'\n"
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
	uint32_t ray_depth = 5;

	float cam_pitch = 0.0f;
	float cam_yaw = 0.0f;
	vec3_t cam_pos = vec3(0.0f, 2.0f, 16.0f);
	float fov = 27.95f; /* 50mm focal lengh */
	vec3_t backcolor = vec3(0.f, 0.f, 0.f);

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
		case OPT_BACKCOLOR: {
			uint32_t color;
			ret = sscanf(optarg, "%x", &color);
			if (ret != 1) {
				fprintf(stderr, "Invalid --backcolor, should be hex.\n");
				return -EINVAL;
			}
			backcolor.x = ((color>>16) & 0xff) / 255.0f;
			backcolor.y = ((color>>8) & 0xff) / 255.0f;
			backcolor.z = (color & 0xff) / 255.0f;
			break;
		}
		case OPT_RAY_DEPTH: {
			ret = sscanf(optarg, "%u", &ray_depth);
			if (ret != 1) {
				fprintf(stderr, "Invalid --ray-depth, unsigned int.\n");
				return -EINVAL;
			}
			break;
		}

		case OPT_LIGHT:
			/* See lights_create() */
			break;
		case OPT_OBJECT:
			/* See objects_create() */
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
	scene = scene_create(opencl, one_frame, width, height, cam_pos, cam_pitch,
			     cam_yaw, fov, backcolor, ray_depth);
	assert(scene);

	/* Init default objects */
	ret = objects_create(scene, argc, argv);
	if (ret)
		goto out;

	/* Init default lights */
	ret = lights_create(scene, argc, argv);
	if (ret)
		goto out;

	/* Commit all scene changes before rendering */
	ret = scene_finish(scene);
	assert(!ret);

	if (one_frame)
		one_frame_render(scene);
	else
		render(scene);

out:
	scene_destroy(scene);
	opencl_deinit(opencl);

	return ret;
}

#endif /* !__OPENCL__ */
