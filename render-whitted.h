#ifndef RENDER_WHITTED_H
#define RENDER_WHITTED_H

#include "render-common.h"

__accelerated static inline bool __ray_cast(struct ray_cast_input *in, struct ray_cast_output *out,
			      __global struct ray_cast_state *s,
			      __global struct octant_queue_entry *q_entries, uint32_t q_depth)
{
	struct intersection isect;

	vec3_t hit_point, hit_normal, hit_color, dir = in->dir;
	vec3_t refract_color, reflect_color;
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

	hit = ray_trace(in->scene, &in->orig, &dir, &isect, PRIMARY_RAY, q_entries, q_depth);
	if (!hit) {
		out->color = scene_sky_color(in->scene, &dir);
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
					       &isect_shadow, SHADOW_RAY, q_entries, q_depth);
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
		vec3_t reflect_orig, reflect_dir, bias;
		refract_color = vec3(0.0f, 0.0f, 0.0f);
		reflect_color = vec3(0.0f, 0.0f, 0.0f);
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

__accelerated static inline vec3_t ray_cast(__global struct scene *scene,
			      __global struct ray_cast_state *ray_states,
			      const vec3_t *orig, const vec3_t *dir,
			      __global struct octant_queue_entry *q_entries, uint32_t q_depth)
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
		bool yielded = __ray_cast(&in, &out, s, q_entries, q_depth);
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
			out.color = scene_sky_color(scene, &in.dir);

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
	return scene_sky_color(scene, orig);
}

__accelerated static inline vec3_t ray_cast_for_pixel(__global struct scene *scene,
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

		vec3_t ray_orig = *orig;
		if (scene->use_defocus) {
			/* Halton bases 5 and 7 for lens jitter (2 and 3 used for pixel) */
			float r  = sqrtf(halton_seq(n, 5));
			float th = 2.0f * M_PI * halton_seq(n, 7);
			vec3_t offset = v3_add(
				v3_muls(scene->defocus_disk_u, r * cosf(th)),
				v3_muls(scene->defocus_disk_v, r * sinf(th)));
			vec3_t focus_pt = v3_add(*orig,
				v3_muls(dir, scene->focus_dist));
			ray_orig = v3_add(*orig, offset);
			dir = v3_norm(v3_sub(focus_pt, ray_orig));
		}

		ray_states_off = (iy * scene->width + ix) * scene->ray_depth;
		ray_states = scene->ray_states + ray_states_off;
		color = v3_add(color, ray_cast(scene, ray_states, &ray_orig, &dir,
					       scene->bvh_queue + (uint64_t)(iy * scene->width + ix) * scene->octant_queue_depth,
					       scene->octant_queue_depth));
	}
	color = v3_divs(color, scene->samples_per_pixel);

	return color;
}

#endif /* RENDER_WHITTED_H */
