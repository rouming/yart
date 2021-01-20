#ifndef SCENE_H
#define SCENE_H

#include "types.h"
#include "math_3d.h"
#include "list.h"
#include "alloc.h"
#include "bvh.h"

#define EPSILON	   1e-5

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

struct opencl;
struct sdl;

enum {
	HEAP_SIZE  = 1<<30,
	CHUNK_SIZE = 1<<10,
};

struct scene {
	uint32_t width;
	uint32_t height;
	float	 fov;
	vec3_t	 backcolor;
	mat4_t	 c2w;
	float	 bias;
	uint32_t ray_depth;
	uint32_t samples_per_pixel;
	uint64_t num_verts;
	struct camera cam;
	__global struct rgba           *framebuffer;
	__global struct ray_cast_state *ray_states;
	__global void    *heap;
	struct allocator alloc;
	struct bvhtree bvhtree;
	struct opencl *opencl;
	struct sdl    *sdl;

	struct list_head mesh_objects;
	struct list_head notmesh_objects;
	struct list_head lights;

	struct {
		uint64_t rays;
	} stat;
};

enum {
	RAY_CAST_CALL,
	RAY_CAST_REFLECT_YIELD,
	RAY_CAST_RR_REFRACT_YIELD,
	RAY_CAST_RR_REFLECT_YIELD,
};

struct ray_cast_state {
	uint32_t type;
	union {
		struct {
			vec3_t        hit_color;
			struct object *hit_object;
		} reflect;
		struct {
			float   kr;
			vec3_t  dir;
			vec3_t  hit_normal;
			vec3_t  hit_point;
			vec3_t  bias;
			uint8_t outside;
		} rr_refract;
		struct {
			float  kr;
			vec3_t refract_color;
		} rr_reflect;
	};
};

enum material_type {
	MATERIAL_PHONG,
	MATERIAL_REFLECT,
	MATERIAL_REFLECT_REFRACT,
};

enum pattern_type {
	PATTERN_UNKNOWN,
	PATTERN_CHECK,
	PATTERN_LINE
};

struct pattern {
	enum pattern_type type;
	float scale;
	float angle;
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
};

enum object_type {
	UNKNOWN_OBJECT = 0,
	SPHERE_OBJECT,
	PLANE_OBJECT,
	MESH_OBJECT,
};

struct object {
	uint32_t          type;  /* object type */
	struct object_ops ops;	 /* because of opencl can't be a pointer */
	struct list_head entry;
	mat4_t o2w;
	enum material_type material;
	struct pattern pattern;
	float  albedo;
	float  ior; /* index of refraction */
	vec3_t Kd;  /* diffuse weight for each RGB channel */
	vec3_t Ks;  /* specular weight for each RGB channel */
	float  n;   /* specular exponent */
	float  r;   /* reflection coef */
};

struct sphere {
	struct object obj;
	float radius;
	float radius_pow2;
	vec3_t center;
};

struct plane {
	struct object obj;
	/*
	 * Components of generic plane equation: Ax + By + Cy + d = 0,
	 * where normal = (A, B, C)
	 */
	vec3_t normal;
	float  d;

	/* Basis on 3D plane to make UV mapping */
	vec3_t b1;
	vec3_t b2;
};

struct triangle_mesh {
	struct object	  obj;
	bool		  smooth_shading; /* smooth shading */
	uint32_t	  num_verts;	  /* number of vertices */
	__global vec3_t	  *vertices;	  /* vertex positions */
	__global vec3_t	  *normals;	  /* vertex normals */
	__global vec2_t	  *sts;		  /* texture coordinates */
};

struct light;

struct light_ops {
	void (*destroy)(struct light *light);
	int (*unmap)(struct light *light);
	void (*illuminate)(__global struct light *light, const vec3_t *orig,
			   vec3_t *dir, vec3_t *intensity, float *distance);
};

enum light_type {
	UNKNOWN_LIGHT = 0,
	DISTANT_LIGHT,
	POINT_LIGHT,
};

struct light {
	uint32_t         type;  /* light type */
	struct light_ops ops;	/* because of opencl can't a pointer */
	struct list_head entry;
	vec3_t color;
	float intensity;
};

struct distant_light {
	struct light light;
	vec3_t dir;
};

struct point_light {
	struct light light;
	vec3_t pos;
};

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

struct ray_cast_input {
	__global struct scene *scene;
	vec3_t orig;
	vec3_t dir;
};

struct ray_cast_output {
	vec3_t color;
};

#endif /* SCENE_H */
