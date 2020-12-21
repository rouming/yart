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
#include <time.h>
#include <errno.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

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
typedef int          int32_t;

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

struct opencl_ctx;

struct scene {
	uint32_t width;
	uint32_t height;
	float    fov;
	vec3_t   back_color;
	mat4_t   c2w;
	float    bias;
	uint32_t max_depth;
	struct opencl_ctx *opencl_ctx;

	struct list_head objects;
	struct list_head lights;
};

#ifndef __OPENCL__

struct opencl_ctx {
	cl_context       context;
	cl_device_id     device_id;
	cl_command_queue queue;
	cl_program       program;
	cl_kernel        kernel;
	cl_mem           scene_dev;
	uint32_t         global_item_size;
};

static inline void *allocate(struct scene *scene, size_t sz)
{
	void *ptr;

	sz += 8;

	if (scene->opencl_ctx) {
		ptr = clSVMAlloc(scene->opencl_ctx->context,
				 CL_MEM_READ_WRITE, sz, 0);
	} else {
		ptr = malloc(sz);
	}
	if (!ptr)
		return ptr;

	*(struct scene **)ptr = scene;

	return ptr + 8;
}

static inline void destroy(void *ptr)
{
	struct scene *scene;

	ptr -= 8;
	scene = *(struct scene **)ptr;
	if (scene->opencl_ctx) {
		clSVMFree(scene->opencl_ctx->context, ptr);
	} else {
		free(ptr);
	}
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

#endif

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

enum material_type {
	MATERIAL_PHONG
};

struct object;

struct object_ops {
	void (*destroy)(struct object *obj);
	bool (*intersect)(struct object *obj, const vec3_t *orig, const vec3_t *dir,
			  float *near, uint32_t *index, vec2_t *uv);
	void (*get_surface_props)(struct object *obj, const vec3_t *hit_point,
				  const vec3_t *dir, uint32_t index, const vec2_t *uv,
				  vec3_t *hit_normal,
				  vec2_t *hit_tex_coords);
};

struct object {
	struct object_ops ops;   /* because of opencl can't be a pointer */
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

static bool sphere_intersect(struct object *obj, const vec3_t *orig,
			     const vec3_t *dir, float *near, uint32_t *index,
			     vec2_t *uv)
{
	struct sphere *sphere = container_of(obj, struct sphere, obj);

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

static void sphere_get_surface_props(struct object *obj, const vec3_t *hit_point,
				     const vec3_t *dir, uint32_t index,
				     const vec2_t *uv, vec3_t *hit_normal,
				     vec2_t *hit_tex_coords)
{
	struct sphere *sphere = container_of(obj, struct sphere, obj);

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
	struct object obj;
	uint32_t num_tris;       /* number of triangles */
	vec3_t   *P;             /* triangles vertex position */
	uint32_t *tris_index;    /* vertex index array */
	vec3_t   *N;             /* triangles vertex normals */
	vec2_t   *sts;           /* triangles texture coordinates */
	bool     smooth_shading; /* smooth shading */
};

static bool triangle_intersect(const vec3_t *orig, const vec3_t *dir,
			       const vec3_t *v0, const vec3_t *v1, const vec3_t *v2,
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

static bool triangle_mesh_intersect(struct object *obj, const vec3_t *orig,
				    const vec3_t *dir, float *near, uint32_t *index,
				    vec2_t *uv)
{
	struct triangle_mesh *mesh =
		container_of(obj, struct triangle_mesh, obj);

	uint32_t j, i;
        bool isect;

	isect = false;
        for (i = 0, j = 0; i < mesh->num_tris; i++) {
		const vec3_t *P = mesh->P;
		const vec3_t *v0 = &P[mesh->tris_index[j + 0]];
		const vec3_t *v1 = &P[mesh->tris_index[j + 1]];
		const vec3_t *v2 = &P[mesh->tris_index[j + 2]];
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

static void triangle_mesh_get_surface_props(struct object *obj, const vec3_t *hit_point,
					    const vec3_t *dir, uint32_t index,
					    const vec2_t *uv, vec3_t *hit_normal,
					    vec2_t *hit_tex_coords)
{
	struct triangle_mesh *mesh =
		container_of(obj, struct triangle_mesh, obj);

	vec2_t st0, st1, st2;
	const vec2_t *sts;

	if (mesh->smooth_shading) {
		/* vertex normal */
		const vec3_t *N = mesh->N;
		vec3_t n0 = N[index * 3 + 0];
		vec3_t n1 = N[index * 3 + 1];
		vec3_t n2 = N[index * 3 + 2];

		n0 = v3_muls(n0, 1 - uv->x - uv->y);
		n1 = v3_muls(n1, uv->x);
		n1 = v3_muls(n2, uv->y);

		*hit_normal = v3_add(n2, v3_add(n0, n1));
        }
        else {
		/* face normal */
		const vec3_t *P = mesh->P;
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
	void (*illuminate)(struct light *light, const vec3_t *orig,
			   vec3_t *dir, vec3_t *intensity, float *distance);
};

struct light {
	struct light_ops ops;   /* because of opencl can't a pointer */
	struct list_head entry;
	vec3_t color;
	float intensity;
	mat4_t l2w;
};

struct distant_light {
	struct light light;
	vec3_t dir;
};

static void distant_light_illuminate(struct light *light, const vec3_t *orig,
				     vec3_t *dir, vec3_t *intensity, float *distance)
{
	struct distant_light *dlight =
		container_of(light, struct distant_light, light);

	*dir = dlight->dir;
	*intensity = v3_muls(dlight->light.color, dlight->light.intensity);
        *distance = INFINITY;
}

struct point_light {
	struct light light;
	vec3_t pos;
};

static void point_light_illuminate(struct light *light, const vec3_t *orig,
				   vec3_t *dir, vec3_t *intensity, float *distance)
{
	struct point_light *plight =
		container_of(light, struct point_light, light);
	float r_pow2;

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

enum ray_type {
	PRIMARY_RAY,
	SHADOW_RAY
};

struct intersection {
	struct object *hit_object;
	float near;
	vec2_t uv;
	uint32_t index;
};

static bool trace(struct scene *scene, const vec3_t *orig, const vec3_t *dir,
		  struct intersection *isect, enum ray_type ray_type)
{
	struct object *obj;

	isect->hit_object = NULL;
	isect->near = INFINITY;

	list_for_each_entry(obj, &scene->objects, entry) {
		float near = INFINITY;
		uint32_t index = 0;
		vec2_t uv;

		if (obj->ops.intersect(obj, orig, dir, &near, &index, &uv) &&
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

static vec3_t cast_ray(struct scene *scene, const vec3_t *orig, const vec3_t *dir,
		       uint32_t depth)
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
		isect.hit_object->ops.get_surface_props(isect.hit_object,
							&hit_point, dir,
							isect.index,
							&isect.uv,
							&hit_normal,
							&hit_tex_coords);
		switch (isect.hit_object->material) {
		case MATERIAL_PHONG: {
			/*
			 * Light loop (loop over all lights in the scene
			 * and accumulate their contribution)
			 */
			vec3_t diffuse, specular;
			struct light *light;

			diffuse = specular = vec3(0.0f, 0.0f, 0.0f);

			list_for_each_entry(light, &scene->lights, entry) {
				vec3_t light_dir, light_intensity;
				vec3_t point, rev_light_dir, R;
				vec3_t rev_dir, diff, spec;

				struct intersection isect_shadow;
				float near, p;
				bool obstacle;

				light->ops.illuminate(light, &hit_point, &light_dir,
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

#ifdef __OPENCL__

__kernel void render_opencl(__global const struct scene *scene,
			    __global vec3_t *framebuffer,
			    __global struct object *objects,
			    __global struct light *lights)
{
	float x, y, scale, img_ratio;
	vec3_t orig, dir;
	uint32_t i, ix, iy;

	scale = tan(deg2rad(scene->fov * 0.5));
	img_ratio = scene->width / (float)scene->height;

	/* Camera position */
	orig = m4_mul_pos(scene->c2w, vec3(0.f, 0.f, 0.f));

	i = get_global_id(0);
	iy = i / scene->width;
	ix = i % scene->width;

	x = (2 * (ix + 0.5) / (float)scene->width - 1) * img_ratio * scale;
	y = (1 - 2 * (iy + 0.5) / (float)scene->height) * scale;

	dir = m4_mul_dir(scene->c2w, vec3(x, y, -1));
	dir = v3_norm(dir);

	framebuffer[i] = cast_ray(scene, &orig, &dir, 0);
}

#else /* !__OPENCL__ */

static void render(struct scene *scene)
{
	vec3_t *buffer, *pix, orig;
	float scale, img_ratio;
	unsigned long long ns;
	uint32_t i, j;
	FILE *out;

	buffer = pix = calloc(scene->width * scene->height, sizeof(*buffer));
	assert(buffer);

	scale = tan(deg2rad(scene->fov * 0.5));
	img_ratio = scene->width / (float)scene->height;

	/* Camera position */
	orig = m4_mul_pos(scene->c2w, vec3(0.f, 0.f, 0.f));

	ns = nsecs();

	for (j = 0; j < scene->height; ++j) {
		for (i = 0; i < scene->width; ++i) {
			float x = (2 * (i + 0.5) / (float)scene->width - 1) *	img_ratio * scale;
			float y = (1 - 2 * (j + 0.5) / (float)scene->height) * scale;
			vec3_t dir;

			dir = m4_mul_dir(scene->c2w, vec3(x, y, -1));
			dir = v3_norm(dir);

			*pix = cast_ray(scene, &orig, &dir, 0);
			pix++;
		}
		fprintf(stderr, "\r%3d%c", (uint32_t)(j / (float)scene->height * 100), '%');
	}
	fprintf(stderr, "\rDone: %.2f (sec)\n", (nsecs() - ns) / 1000000000.0);

	/* save framebuffer to file */
	out = fopen("yart-out.ppm", "w");
	assert(out);

	fprintf(out, "P6\n%d %d\n255\n", scene->width, scene->height);
	for (i = 0; i < scene->height * scene->width; ++i) {
		char r = (char)(255 * clamp(0.0f, 1.0f, buffer[i].x));
		char g = (char)(255 * clamp(0.0f, 1.0f, buffer[i].y));
		char b = (char)(255 * clamp(0.0f, 1.0f, buffer[i].z));

		fprintf(out, "%c%c%c", r, g, b);
	}
	fclose(out);
	free(buffer);
}

static int opencl_init(struct scene *scene,
		       vec3_t *framebuffer,
		       struct object *objects,
		       struct light *lights,
		       const char *kernel_fn,
		       struct opencl_ctx *ctx)
{
	/* Load current file */
	const char *path = __FILE__;

	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_command_queue queue;
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_mem scene_dev;
	cl_int ret;

	size_t size, read;
	char *source;
	void *buf;
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

	ret = clGetDeviceIDs(platform_id,
			     CL_DEVICE_TYPE_GPU | CL_DEVICE_SVM_CAPABILITIES,
			     1, &device_id, &ret_num_devices);
	assert(!ret);

	/* Create an OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	assert(!ret);

	/* Create a command queue */
	queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
	assert(!ret);

	/* Create memory buffers on the device for each vector */
	buf = clSVMAlloc(context, CL_MEM_WRITE_ONLY,
			 scene->width * scene->height * sizeof(*buf), 0);
	assert(buf);

	/* Create a program from the kernel source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source,
					    &size, &ret);
	assert(!ret);

	free(source);

	/* Build the program */
	ret = clBuildProgram(program, 1, &device_id, "-Werror -D__OPENCL__",
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
		printf("%s\n", log);
		free(log);

		return -1;
	} else if (ret) {
		printf("clBuildProgram: failed %d\n", ret);
		return -1;
	}

	/* Create the opencl kernel */
	kernel = clCreateKernel(program, kernel_fn, &ret);
	assert(!ret);

	/* Map a scene pointer to the device */
	scene_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
				   sizeof(*scene), scene, &ret);
	assert(!ret);

	/* Set the arguments of the kernel */
	ret = clSetKernelArg(kernel, 0, sizeof(scene_dev), &scene_dev);
	assert(!ret);
	ret = clSetKernelArgSVMPointer(kernel, 1, framebuffer);
	assert(!ret);
	ret = clSetKernelArgSVMPointer(kernel, 2, objects);
	assert(!ret);
	ret = clSetKernelArgSVMPointer(kernel, 3, lights);
	assert(!ret);

	/* Init context */
	ctx->scene_dev = scene_dev;
	ctx->context = context;
	ctx->device_id = device_id;
	ctx->queue = queue;
	ctx->program = program;
	ctx->kernel = kernel;
	ctx->global_item_size = scene->width * scene->height;

	scene->opencl_ctx = ctx;

	return 0;
}

static void opencl_deinit(struct opencl_ctx *ctx)
{
	if (!ctx)
		return;

	clReleaseMemObject(ctx->scene_dev);
	clReleaseKernel(ctx->kernel);
	clReleaseProgram(ctx->program);
	clReleaseCommandQueue(ctx->queue);
	clReleaseContext(ctx->context);
}

static int opencl_invoke(struct opencl_ctx *ctx)
{
	size_t global_item_size = ctx->global_item_size;
	/* Divide work items into groups of 64 */
	size_t local_item_size = 64;

	return clEnqueueNDRangeKernel(ctx->queue, ctx->kernel, 1, NULL,
				      &global_item_size,
				      &local_item_size, 0,
				      NULL, NULL);
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
	if (obj->ops.destroy)
		obj->ops.destroy(obj);
}

struct object_ops sphere_ops = {
	.intersect         = sphere_intersect,
	.get_surface_props = sphere_get_surface_props
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

	destroy(mesh->P);
	destroy(mesh->tris_index);
	destroy(mesh->N);
	destroy(mesh->sts);
	destroy(mesh);
}

struct object_ops triangle_mesh_ops = {
	.destroy           = triangle_mesh_destroy,
	.intersect         = triangle_mesh_intersect,
	.get_surface_props = triangle_mesh_get_surface_props
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
        P = allocate(scene, max_vert_index * sizeof(*P));
	assert(P);

	/* Transform vertices to world space */
        for (i = 0; i < max_vert_index; ++i)
		P[i] = m4_mul_pos(*o2w, verts[i]);

	tris_index = allocate(scene, num_tris * 3 * sizeof(*tris_index));
	assert(tris_index);

	N = allocate(scene, num_tris * 3 * sizeof(*N));
	assert(N);

	sts = allocate(scene, num_tris * 3 * sizeof(*sts));
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

	mesh = allocate(scene, sizeof(*mesh));
	assert(mesh);

	f = fopen(file, "r");
	assert(f);

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

	normals = calloc(verts_arr_sz, sizeof(*normals));
	assert(normals);

	for (i = 0; i < verts_arr_sz; i++) {
		vec3_t *norm = &normals[i];
		ret = fscanf(f, "%f%f%f", &norm->x, &norm->y, &norm->z);
		assert(ret == 3);
	}

	st = calloc(verts_arr_sz, sizeof(*st));
	assert(st);

	for (i = 0; i < verts_arr_sz; i++) {
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

		sphere = allocate(scene, sizeof(*sphere));
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

struct light_ops distant_light_ops = {
	.illuminate = distant_light_illuminate
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

__attribute__((unused))
static void point_light_init(struct point_light *plight, const mat4_t *l2w,
			     const vec3_t *color, float intensity)
{
	struct light_ops point_light_ops = {
		.illuminate = point_light_illuminate
	};
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

	dlight = allocate(scene, sizeof(*dlight));
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
		destroy(light);
	}
}

static void scene_init(struct scene *scene)
{
	*scene = (struct scene) {
		.width        = 640,
		.height       = 480,
		.fov          = 90,
		.back_color   = {0.235294f, 0.67451f, 0.843137f},
		.c2w          = m4_identity(),
		.bias         = 0.0001,
		.max_depth    = 5,
		.objects      = LIST_HEAD_INIT(scene->objects),
		.lights       = LIST_HEAD_INIT(scene->lights),
	};
};

static void scene_deinit(struct scene *scene)
{
	opencl_deinit(scene->opencl_ctx);
	objects_destroy(scene);
	lights_destroy(scene);
}

int main(int argc, char **argv)
{
	bool use_opencl = false;
	int ret;

	struct opencl_ctx ctx;
	struct scene scene;

	if (argc > 1)
		use_opencl = !strcmp("opencl", argv[1]);

	/* Init scene */
	scene_init(&scene);
	scene.fov = 36.87;
	scene.width = 1024;
	scene.height = 747;

	/* Camera position */
	scene.c2w.m32 = 12;
	scene.c2w.m31 = 1;

	if (use_opencl) {
		/* Init opencl context */
		ret = opencl_init(&scene, NULL, NULL, NULL,
				  "render_opencl", &ctx);
		if (ret)
			return -1;
	}

	/* Init default objects */
	ret = objects_create(&scene);
	assert(!ret);

	/* Init default light */
	lights_create(&scene);

	/* Finally render */
	render(&scene);

	scene_deinit(&scene);

	return 0;
}

#endif /* !__OPENCL__ */
