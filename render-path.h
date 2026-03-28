#ifndef RENDER_PATH_H
#define RENDER_PATH_H

#include "render-common.h"

/*
 * XOR-shift32 PRNG -- fast, no global state, GPU-safe.
 * Each pixel/sample gets its own seed via path_rng_seed().
 */
__accelerated static inline uint32_t path_rng_next(uint32_t *state)
{
	*state ^= *state << 13;
	*state ^= *state >> 17;
	*state ^= *state << 5;
	return *state;
}

/* Returns a float in [0, 1) */
__accelerated static inline float path_rng_float(uint32_t *state)
{
	return (path_rng_next(state) & 0xFFFFFFu) * (1.0f / 0x1000000u);
}

/* Wang hash -- mixes pixel index and sample number into a good seed */
__accelerated static inline uint32_t path_rng_seed(uint32_t pixel, uint32_t sample)
{
	uint32_t s = pixel * 1973u + sample * 9277u + 12345u;

	s = (s ^ 61u) ^ (s >> 16u);
	s *= 9u;
	s ^= s >> 4u;
	s *= 0x27d4eb2du;
	s ^= s >> 15u;
	return s;
}

/*
 * Build an orthonormal basis (tangent, bitangent) around a normal.
 * Used to transform local hemisphere samples to world space.
 */
__accelerated static inline void
path_build_onb(const vec3_t *n, vec3_t *t, vec3_t *b)
{
	vec3_t up = (fabsf(n->x) < 0.9f) ? vec3(1.0f, 0.0f, 0.0f)
	                                  : vec3(0.0f, 1.0f, 0.0f);
	*t = v3_norm(v3_cross(up, *n));
	*b = v3_cross(*n, *t);
}

/* Random point on the unit sphere -- used for mirror fuzz perturbation */
__accelerated static inline vec3_t path_random_on_sphere(uint32_t *rng)
{
	float theta = 2.0f * M_PI * path_rng_float(rng);
	float phi   = acosf(1.0f - 2.0f * path_rng_float(rng));

	return vec3(sinf(phi) * cosf(theta),
	            sinf(phi) * sinf(theta),
	            cosf(phi));
}

/*
 * Cosine-weighted hemisphere sample around 'normal'.
 * PDF = cos(theta)/pi; for lambertian BRDF = Kd/pi the
 * sample weight simplifies to just Kd -- no explicit PDF division needed.
 */
__accelerated static inline vec3_t
path_cosine_sample(const vec3_t *normal, uint32_t *rng)
{
	float r1  = path_rng_float(rng);
	float r2  = path_rng_float(rng);
	float phi = 2.0f * M_PI * r1;
	float sin_theta = sqrtf(r2);
	float cos_theta = sqrtf(1.0f - r2);

	vec3_t t, b;

	path_build_onb(normal, &t, &b);

	return v3_add(v3_muls(t,       cosf(phi) * sin_theta),
	       v3_add(v3_muls(b,       sinf(phi) * sin_theta),
	              v3_muls(*normal, cos_theta)));
}

/*
 * Compute scatter direction and attenuation for a ray hitting an object.
 * Returns false if the ray is absorbed (no further bouncing).
 */
__accelerated static inline bool
path_scatter(__global struct object *obj, const vec3_t *dir,
             const vec3_t *hit_normal, uint32_t *rng,
             vec3_t *attenuation, vec3_t *scatter_dir)
{
	switch (obj->material) {
	case MATERIAL_LAMBERTIAN: {
		*scatter_dir = path_cosine_sample(hit_normal, rng);
		*attenuation = obj->Kd;
		return true;
	}
	case MATERIAL_MIRROR: {
		vec3_t r = reflect(dir, hit_normal);

		if (obj->fuzz > 0.0f)
			r = v3_add(r, v3_muls(path_random_on_sphere(rng), obj->fuzz));
		*scatter_dir = v3_norm(r);
		*attenuation = obj->Kd;
		/* Absorbed if fuzz kicked the ray below the surface */
		return v3_dot(*scatter_dir, *hit_normal) > 0.0f;
	}
	case MATERIAL_DIELECTRIC: {
		float kr = fresnel(dir, hit_normal, obj->ior);

		/*
		 * Choose refraction or reflection stochastically based on
		 * Fresnel reflectance.  kr == 1 (total internal reflection)
		 * always falls through to reflect.
		 */
		if (path_rng_float(rng) > kr) {
			vec3_t rd = refract(dir, hit_normal, obj->ior);

			if (v3_dot(rd, rd) > EPSILON) {
				*scatter_dir = v3_norm(rd);
				*attenuation = vec3(1.0f, 1.0f, 1.0f);
				return true;
			}
		}
		*scatter_dir = reflect(dir, hit_normal);
		*attenuation = vec3(1.0f, 1.0f, 1.0f);
		return true;
	}
	default:
		return false;
	}
}

/*
 * Trace a single path from (orig, dir) and return the accumulated radiance.
 * The loop is flat -- no recursion, no yield/resume state machine needed.
 * Terminates on: sky miss, ray absorbed by material, Russian Roulette, or
 * max depth (scene->ray_depth).
 */
__accelerated static inline vec3_t
path_cast(__global struct scene *scene, const vec3_t *orig, const vec3_t *dir,
          __global struct octant_queue_entry *q_entries, uint32_t q_depth,
          uint32_t *rng)
{
	vec3_t attenuation = vec3(1.0f, 1.0f, 1.0f);
	vec3_t ray_orig    = *orig;
	vec3_t ray_dir     = *dir;
	int depth;

	for (depth = 0; depth < scene->ray_depth; depth++) {
		struct intersection isect;
		vec3_t hit_point, hit_normal, scatter_dir, attn;
		vec2_t hit_tex_coords;

		atomic64_inc(&scene->stat.rays);

		if (!ray_trace(scene, &ray_orig, &ray_dir, &isect, PRIMARY_RAY,
		               q_entries, q_depth)) {
			/*
			 * Ray escaped to the sky -- backcolor is the light source.
			 * All accumulated attenuation flows back through the path.
			 */
			return v3_mul(attenuation, scene_sky_color(scene, &ray_dir));
		}

		hit_point = v3_add(ray_orig, v3_muls(ray_dir, isect.near));
		object_get_surface_props(isect.hit_object, &hit_point, &ray_dir,
		                         isect.index, &isect.uv,
		                         &hit_normal, &hit_tex_coords);

		/*
		 * Russian Roulette: after a few bounces start randomly
		 * terminating paths whose attenuation has fallen low.
		 * Surviving paths are boosted to keep the estimator unbiased.
		 */
		if (depth > 2) {
			float p = MAX(attenuation.x,
			              MAX(attenuation.y, attenuation.z));
			if (path_rng_float(rng) > p)
				return vec3(0.0f, 0.0f, 0.0f);
			attenuation = v3_divs(attenuation, p);
		}

		if (!path_scatter(isect.hit_object, &ray_dir, &hit_normal, rng,
		                  &attn, &scatter_dir))
			return vec3(0.0f, 0.0f, 0.0f);

		attenuation = v3_mul(attenuation, attn);

		/*
		 * Offset origin along the normal to avoid self-intersection.
		 * Use the scatter direction to determine which side to offset to.
		 */
		if (v3_dot(scatter_dir, hit_normal) > 0.0f)
			ray_orig = v3_add(hit_point, v3_muls(hit_normal, scene->bias));
		else
			ray_orig = v3_sub(hit_point, v3_muls(hit_normal, scene->bias));

		ray_dir = scatter_dir;
	}

	/* Exhausted max depth */
	return vec3(0.0f, 0.0f, 0.0f);
}

/*
 * Per-pixel entry point -- same signature as ray_cast_for_pixel so the
 * runtime switch in the render kernel is a straight drop-in.
 */
__accelerated static inline vec3_t
path_cast_for_pixel(__global struct scene *scene, const vec3_t *orig,
                    int ix, int iy, float scale, float img_ratio)
{
	vec3_t color = vec3(0.0f, 0.0f, 0.0f);
	int n;

	for (n = 1; n <= scene->samples_per_pixel; n++) {
		uint32_t rng = path_rng_seed((uint32_t)(iy * scene->width + ix),
		                              (uint32_t)n);
		vec3_t dir;
		float x, y;

		/* Random jitter within the pixel for anti-aliasing */
		x = ix + path_rng_float(&rng);
		y = iy + path_rng_float(&rng);

		x = (2.0f * x / scene->width  - 1.0f) * img_ratio * scale;
		y = (1.0f - 2.0f * y / scene->height) * scale;

		dir = m4_mul_dir(scene->c2w, vec3(x, y, -1.0f));
		dir = v3_norm(dir);

		vec3_t ray_orig = *orig;
		if (scene->use_defocus) {
			/*
			 * Thin-lens defocus: jitter the ray origin on the lens
			 * disk, redirect toward the focus point.  Disk sample
			 * uses polar mapping for branch-free uniform coverage.
			 */
			float r  = sqrtf(path_rng_float(&rng));
			float th = 2.0f * M_PI * path_rng_float(&rng);
			vec3_t offset = v3_add(
				v3_muls(scene->defocus_disk_u, r * cosf(th)),
				v3_muls(scene->defocus_disk_v, r * sinf(th)));
			vec3_t focus_pt = v3_add(*orig,
				v3_muls(dir, scene->focus_dist));
			ray_orig = v3_add(*orig, offset);
			dir = v3_norm(v3_sub(focus_pt, ray_orig));
		}

		color = v3_add(color,
		               path_cast(scene, &ray_orig, &dir,
		                         scene->bvh_queue +
		                         (uint64_t)(iy * scene->width + ix) *
		                         scene->octant_queue_depth,
		                         scene->octant_queue_depth, &rng));
	}

	return v3_divs(color, (float)scene->samples_per_pixel);
}

#endif /* RENDER_PATH_H */
