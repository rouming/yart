// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * YART (Yet Another Ray Tracer) boosted by OpenCL
 * Copyright (C) 2020,2021 Roman Penyaev
 *
 * Based on lessons from scratchapixel.com and pbr-book.org
 *
 * Roman Penyaev <r.peniaev@gmail.com>
 */

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

#include "scene.h"
#include "buf.h"
#include "render-opencl.h"

#define MOVE_SPEED 0.03f

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

void *buf_allocate(struct opencl *opencl, size_t sz)
{
	return __buf_allocate(opencl, sz, BUF_ZERO | BUF_MAP_WRITE);
}

void buf_destroy(void *ptr)
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

int buf_map(void *ptr, uint32_t flags)
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

int buf_unmap(void *ptr)
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

static int opencl_init(struct opencl *opencl, const char *kernel_fn)
{
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

	const char *source = (char *)render_opencl_preprocessed_cl;
	size_t size = render_opencl_preprocessed_cl_len;

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
	size = render_opencl_preprocessed_cl_len;
	program = clCreateProgramWithSource(context, 1, &source, &size, &ret);
	assert(!ret);

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
	.get_surface_props	= sphere_get_surface_props,
};

static void plane_destroy(struct object *obj)
{
	struct plane *plane =
		container_of(obj, struct plane, obj);

	buf_destroy(plane);
}

static int plane_unmap(struct object *obj)
{
	struct plane *plane =
		container_of(obj, struct plane, obj);

	return buf_unmap(plane);
}

struct object_ops plane_ops = {
	.destroy		= plane_destroy,
	.unmap			= plane_unmap,
	.intersect		= plane_intersect,
	.get_surface_props	= plane_get_surface_props,
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
	.get_surface_props	= triangle_mesh_get_surface_props,
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
	OPT_SAMPLES_PER_PIXEL,

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
	{"samples-per-pixel",
		      required_argument, 0, OPT_SAMPLES_PER_PIXEL},
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
	OBJECT_PATTERN,
	OBJECT_ALBEDO,
	OBJECT_IOR,
	OBJECT_KD,
	OBJECT_KS,
	OBJECT_N,
	OBJECT_R,
	OBJECT_MESH_FILE,
	OBJECT_MESH_SMOOTH_SHADING,
	OBJECT_SPHERE_RADIUS,
	OBJECT_SPHERE_POS,
	OBJECT_PLANE_NORMAL,
	OBJECT_PLANE_D,
};

static char *const object_token[] = {
	[OBJECT_TYPE]	       = "type",
	[OBJECT_MATERIAL]      = "material",
	[OBJECT_ROTATE_X]      = "rotate-x",
	[OBJECT_ROTATE_Y]      = "rotate-y",
	[OBJECT_ROTATE_Z]      = "rotate-z",
	[OBJECT_SCALE]	       = "scale",
	[OBJECT_TRANSLATE]     = "translate",
	[OBJECT_PATTERN]       = "pattern",
	[OBJECT_ALBEDO]	       = "albedo",
	[OBJECT_IOR]	       = "ior",
	[OBJECT_KD]	       = "Kd",
	[OBJECT_KS]	       = "Ks",
	[OBJECT_N]	       = "n",
	[OBJECT_R]	       = "r",
	[OBJECT_MESH_FILE]     = "file",
	[OBJECT_MESH_SMOOTH_SHADING] = "smooth-shading",
	[OBJECT_SPHERE_RADIUS] = "radius",
	[OBJECT_SPHERE_POS]    = "pos",
	[OBJECT_PLANE_NORMAL]  = "normal",
	[OBJECT_PLANE_D]       = "d",
	NULL
};

struct object_params {
	int    parsed_params_bits;
	mat4_t o2w;
	enum object_type   type;
	enum material_type material;
	struct pattern pattern;
	float  albedo;
	float  ior;
	vec3_t Kd;
	vec3_t Ks;
	float  n;
	float  r;
	struct {
		char  file[512];
		bool  smooth_shading;
	} mesh;
	struct {
		float  radius;
		vec3_t pos;
	} sphere;
	struct {
		vec3_t normal;
		float  d;
	} plane;
};

static void default_object_params(struct object_params *params)
{
	memset(params, 0, sizeof(*params));
	params->material = MATERIAL_PHONG;
	params->pattern = (struct pattern){
		.type = PATTERN_UNKNOWN,
		.scale = 0.5f,
		.angle = 15.0f
	};
	params->albedo = 0.18f;
	params->ior = 1.3f;
	params->Kd = vec3(0.8f, 0.8f, 0.8f);
	params->Ks = vec3(0.2f, 0.2f, 0.2f);
	params->n = 10.0f;
	params->r = 0.0f;
	params->sphere.radius = 0.5f;
	params->sphere.pos = vec3(0.0f, 0.0f, 0.0f);
	params->plane.normal = vec3(0.0f, 1.0f, 0.0f);
	params->plane.d = 0.0f;
	params->o2w = m4_identity();
}

static void object_init(struct object *obj, struct object_ops *ops,
			struct object_params *params)
{
	INIT_LIST_HEAD(&obj->entry);
	obj->type = params->type;
	obj->ops = *ops;
	obj->o2w = params->o2w;
	obj->material = params->material;
	obj->pattern = params->pattern;
	obj->albedo = params->albedo;
	obj->ior = params->ior;
	obj->Kd = params->Kd;
	obj->Ks = params->Ks;
	obj->n = params->n;
	obj->r = params->r;
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

static void plane_init(struct plane *plane, struct object_params *params)
{
	vec3_t up;

	object_init(&plane->obj, &plane_ops, params);
	plane->normal = v3_norm(params->plane.normal);
	plane->d = params->plane.d;

	up = vec3(0.0f, 1.0f, 0.0f);
	if (v3_dot(up, plane->normal) >= 0.999)
		up = vec3(1.0f, 0.0f, 0.0f);

	/* Form two basis vectors */
	plane->b1 = v3_cross(plane->normal, up);
	plane->b2 = v3_cross(plane->normal, plane->b1);
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
	scene->num_verts += mesh->num_verts;
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
		fprintf(stderr, "Can't open %s, aiImportFile failed\n", params->mesh.file);
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
		scene->num_verts += mesh->num_verts;
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
			else if (!strcmp(real_value, "plane"))
				params->type = PLANE_OBJECT;
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
		case OBJECT_KD:
		case OBJECT_KS:
		case OBJECT_SCALE:
		case OBJECT_TRANSLATE: {
			vec3_t vec;

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

			if (c == OBJECT_SCALE || c == OBJECT_TRANSLATE) {
				mat4_t m;

				if (c == OBJECT_SCALE)
					m = m4_scaling(vec);
				else
					m = m4_translation(vec);
				params->o2w = m4_mul(params->o2w, m);
			} else if (c == OBJECT_KD) {
				params->Kd = vec;
			} else if (c == OBJECT_KS) {
				params->Ks = vec;
			}
			break;
		}
		case OBJECT_PATTERN: {
			char *b = strchr(value, '[');
			char *e = strchr(value, ']');
			int len = b ? b - value : strlen(value);

			if ((!b) ^ (!e) || b > e) {
				fprintf(stderr, "Invalid object pattern type '%s', incorrect brackets\n",
					value);
				return -EINVAL;
			}
			if (!strncmp(value, "check", len))
				params->pattern.type = PATTERN_CHECK;
			else if (!strncmp(value, "line", len))
				params->pattern.type = PATTERN_LINE;
			else {
				fprintf(stderr, "Invalid object pattern type '%s'\n",
					real_value);
				return -EINVAL;
			}
			if (b) {
				char *scale = strstr(value, "scale=");
				char *angle = strstr(value, "angle=");

				if (scale) {
					scale += strlen("scale=");
					ret = sscanf(scale, "%f", &params->pattern.scale);
					if (ret != 1) {
						fprintf(stderr, "Invalid scale for pattern\n");
						return -EINVAL;
					}
				}
				if (angle) {
					angle += strlen("angle=");
					ret = sscanf(angle, "%f", &params->pattern.angle);
					if (ret != 1) {
						fprintf(stderr, "Invalid angle for pattern\n");
						return -EINVAL;
					}
				}

				/* To the next param after bracket and skip comma */
				subopts = e + 1;
				if (*subopts == ',')
					subopts += 1;
			}

			break;
		}
		case OBJECT_ALBEDO:
			fptr = &params->albedo;
			break;
		case OBJECT_IOR:
			fptr = &params->ior;
			break;
		case OBJECT_N:
			fptr = &params->n;
			break;
		case OBJECT_R:
			fptr = &params->r;
			break;
		case OBJECT_SPHERE_RADIUS:
			fptr = &params->sphere.radius;
			break;
		case OBJECT_PLANE_D:
			fptr = &params->plane.d;
			break;
		case OBJECT_SPHERE_POS:
		case OBJECT_PLANE_NORMAL: {
			vec3_t vec;

			ret = sscanf(value, "%f,%f,%f%n", &vec.x, &vec.y, &vec.z, &num);
			if (ret != 3) {
				fprintf(stderr, "Invalid object '%s' parameter\n",
					object_token[c]);
				return -EINVAL;
			}
			subopts = value + num;
			if (subopts[0] == ',')
				/* Skip trailing comma */
				subopts += 1;

			if (c == OBJECT_SPHERE_POS) {
				params->sphere.pos = vec;
			} else if (c == OBJECT_PLANE_NORMAL) {
				params->plane.normal = vec;
			}
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
	case PLANE_OBJECT:
		if (is_parsed_object_param(params, OBJECT_MESH_FILE)) {
			fprintf(stderr, "Invalid parameter '%s' for 'plane' object type\n",
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
	case PLANE_OBJECT: {
		struct plane *plane;

		plane = buf_allocate(scene->opencl, sizeof(*plane));
		if (!plane)
			return -ENOMEM;
		plane_init(plane, params);
		list_add_tail(&plane->obj.entry, &scene->objects);
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

static void light_init(struct light *light, enum light_type type,
		       struct light_ops *ops, const vec3_t *color,
		       float intensity)
{
	INIT_LIST_HEAD(&light->entry);
	light->type = type;
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
};

static void distant_light_set_dir(struct distant_light *dlight, vec3_t dir)
{
	dlight->dir = v3_norm(dir);
}

static void distant_light_init(struct distant_light *dlight, const vec3_t *color,
			       float intensity)
{
	light_init(&dlight->light, DISTANT_LIGHT, &distant_light_ops,
		   color, intensity);
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
};

static void point_light_init(struct point_light *plight, const vec3_t *color,
			     float intensity)
{
	light_init(&plight->light, POINT_LIGHT, &point_light_ops,
		   color, intensity);
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
				  vec3_t backcolor, uint32_t ray_depth,
				  uint32_t samples_per_pixel)
{
	struct scene *scene;
	struct rgba *framebuffer;
	struct ray_cast_state *ray_states;
	int ret;

	/* Don't mmap by default */
	framebuffer = __buf_allocate(opencl, width * height * sizeof(*framebuffer), 0);
	assert(framebuffer);
	ray_states = __buf_allocate(opencl, width * height * ray_depth * sizeof(*ray_states), 0);
	assert(ray_states);

	scene = buf_allocate(opencl, sizeof(*scene));
	assert(scene);

	*scene = (struct scene) {
		.width	      = width,
		.height	      = height,
		.fov	      = fov,
		.backcolor    = backcolor,
		.ray_depth    = ray_depth,
		.samples_per_pixel
			      = samples_per_pixel,
		.c2w	      = m4_identity(),
		.bias	      = 0.0001,
		.opencl	      = opencl,
		.framebuffer  = framebuffer,
		.ray_states   = ray_states,
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
	buf_destroy(scene->ray_states);
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
	uint32_t ix, iy;

	scale = tan(deg2rad(scene->fov * 0.5));
	img_ratio = scene->width / (float)scene->height;

	/* Camera position */
	orig = m4_mul_pos(scene->c2w, vec3(0.f, 0.f, 0.f));

	pix = scene->framebuffer;
	for (iy = 0; iy < scene->height; ++iy) {
		for (ix = 0; ix < scene->width; ++ix) {
			color = ray_cast_for_pixel(scene, &orig, ix, iy,
						   scale, img_ratio);
			color_vec_to_rgba32(&color, pix);
			pix++;
		}
	}
}

static int scene_map_before_read(struct scene *scene)
{
	int ret;

	/* Map framebuffer for reading */
	ret = buf_map(scene->framebuffer, BUF_MAP_READ);
	if (ret) {
		fprintf(stderr, "Failed to map framebuffer for reading\n");
		return ret;
	}

	/* Map stats */
	ret = __buf_map(scene->opencl, &scene->stat, sizeof(scene->stat),
			BUF_MAP_READ);
	if (ret) {
		fprintf(stderr, "Failed to map stat for reading\n");
		buf_unmap(scene->framebuffer);
		return ret;
	}

	return 0;
}

static void scene_unmap_after_read(struct scene *scene)
{
	int ret;

	ret = __buf_unmap(scene->opencl, &scene->stat);
	assert(!ret);
	ret = buf_unmap(scene->framebuffer);
	assert(!ret);
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

	/* Map before read */
	ret = scene_map_before_read(scene);
	assert(!ret);

	fprintf(out, "P6\n%d %d\n255\n", scene->width, scene->height);
	for (i = 0; i < scene->height * scene->width; ++i) {
		struct rgba *rgb = &scene->framebuffer[i];

		fprintf(out, "%c%c%c", rgb->r, rgb->g, rgb->b);
	}
	fclose(out);

	/* Unmap */
	scene_unmap_after_read(scene);
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

static const char *scene_average_rays(struct scene *scene)
{
	static struct welford_state s;
	static uint64_t render_ns;
	static char buf[32];
	static uint64_t rays;
	float rps;

	if (render_ns) {
		rps = 1000000000.0f / (nsecs() - render_ns) *
			(scene->stat.rays - rays);
		rps = avg_welford(&s, rps);

		if (rps > 1e6)
			snprintf(buf, sizeof(buf), "%5.0fM", rps/1e6);
		else if (rps > 1e3)
			snprintf(buf, sizeof(buf), "%5.0fK", rps/1e3);
		else
			snprintf(buf, sizeof(buf), "%5.0f", rps);

	} else {
		snprintf(buf, sizeof(buf), "0");
	}
	render_ns = nsecs();
	rays = scene->stat.rays;

	return buf;
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
	r.h = 140;

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

	snprintf(buf, sizeof(buf), "  Rays %s", scene_average_rays(scene));
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 85, text_surface->w, text_surface->h};
	SDL_BlitSurface(text_surface, NULL, rect_surface, &rr);
	SDL_FreeSurface(text_surface);

	snprintf(buf, sizeof(buf), "  FPS %6.0f", scene_average_fps(scene));
	text_surface = TTF_RenderText_Solid(font, buf, color);
	assert(text_surface);
	rr = (SDL_Rect){0, 100, text_surface->w, text_surface->h};
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

		/* Map before read */
		ret = scene_map_before_read(scene);
		assert(!ret);

		SDL_UpdateTexture(sdl->screen, NULL, scene->framebuffer,
				  scene->width * sizeof(*scene->framebuffer));

		SDL_RenderCopy(sdl->renderer, sdl->screen, NULL, NULL);
		draw_scene_status(scene);
		SDL_RenderPresent(sdl->renderer);

		/* Unmap */
		scene_unmap_after_read(scene);
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
	       "   --samples-per-pixel\n"
	       "               - multisample anti-aliasing technique, number of samples (rays) per a pixel, 4 is default\n"
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
	       "                 'type'      - required parameter, specifies type of the object, 'mesh', 'sphere' or 'plane'\n"
	       "                               can be specified\n"
	       "                 'material'  - object material (shading), should be 'phong', 'reflect', 'reflect-refract'\n"
	       "                 'rotate-x'\n"
	       "                 'rotate-y'\n"
	       "                 'rotate-z'  - rotate around axis by a give angle in degrees\n"
	       "                 'scale'     - scale on specified vector, accepts a single float or float,float,float\n"
	       "                 'translate' - translates on specified offset vector, accepts float,float,float\n"
	       "                 'pattern'   - apply pattern on object using UV coordinates, should be 'check' or 'line'\n"
	       "                               pattern can be scaled or rotated by providing parameters in square brackets, e.g.:\n"
	       "                               pattern=check[scale=0.5,angle=10]\n"
	       "                 'albedo' - albedo\n"
	       "                 'ior'    - index of refraction\n"
	       "                 'Kd'     - diffuse weight, accepts float or float,float,float\n"
	       "                 'Ks'     - specular weight, accepts float or float,float,float\n"
	       "                 'n'      - specular exponent\n"
	       "                 'r'      - reflection coefficient, accepts float\n"
	       "                Sphere:\n"
	       "                 'radius' - sphere radius\n"
	       "                 'pos'    - spehere position\n"
	       "                Plane:\n"
	       "                 'normal' - plane normal, accepts float,float,float\n"
	       "                 'd'      - plane offset, d component of plane equation\n"
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
	uint32_t samples_per_pixel = 4;

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
			if (!ray_depth) {
				fprintf(stderr, "Invalid ray depth value.\n");
				return -EINVAL;
			}
			break;
		}
		case OPT_SAMPLES_PER_PIXEL: {
			ret = sscanf(optarg, "%u", &samples_per_pixel);
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
			     cam_yaw, fov, backcolor, ray_depth, samples_per_pixel);
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
