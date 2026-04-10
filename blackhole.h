#ifndef BLACKHOLE_H
#define BLACKHOLE_H

#include "render-common.h"

/*
 * blackhole_gravity -- sum gravitational acceleration on a ray at 'pos'
 * from all black holes in the scene.
 *
 * Derivation:
 *   Newton's law of universal gravitation: F = G*M*m / r^2
 *   For a photon treated as a unit-mass test particle (m = 1) with G = 1:
 *     a = M / r^2   (magnitude)
 *
 *   Direction: unit vector pointing from the ray toward the black hole center.
 *   Combined: a_vec = M * (center - pos) / |center - pos|^3
 *     (the extra /r turns the direction vector into a unit vector)
 *
 *   For N black holes, accelerations add by the superposition principle --
 *   each black hole contributes independently.
 *
 * Note: this is the Newtonian approximation. It underestimates deflection
 * by a factor of 2 compared to GR at large distances, but captures the
 * qualitative photon-orbit and event-horizon behaviour well for visualization.
 */
__accelerated static inline vec3_t
blackhole_gravity(__global struct scene *scene, const vec3_t *pos)
{
	__global struct object *obj;
	vec3_t acc = vec3(0.0f, 0.0f, 0.0f);

	list_for_each_entry(obj, &scene->notmesh_objects, entry) {
		__global struct blackhole *bh;
		vec3_t to_bh;
		float r2, r, a;

		if (obj->type != BLACKHOLE_OBJECT)
			continue;

		bh = container_of(obj, typeof(*bh), obj);

		/*
		 * Vector from the current ray position to the black hole center.
		 * This is the direction in which gravity pulls the ray.
		 */
		to_bh = v3_sub(bh->center, *pos);

		/*
		 * r^2 = dot(to_bh, to_bh) = squared Euclidean distance.
		 * Guard against division by zero if the ray reaches the center.
		 */
		r2 = v3_dot(to_bh, to_bh);
		if (r2 < EPSILON)
			continue;

		r = sqrtf(r2);

		/*
		 * Acceleration magnitude: a = G*M / r^2 = M / r^2  (G = 1).
		 *
		 * Full vector: acc_vec = a * (to_bh / r) = M * to_bh / r^3
		 * We compute it as v3_muls(to_bh, M / (r^2 * r)) = v3_muls(to_bh, a/r).
		 */
		a = bh->mass / r2;
		acc = v3_add(acc, v3_muls(to_bh, a / r));
	}

	return acc;
}

/*
 * bh_potential -- Newtonian gravitational potential at 'pos' from all BHs.
 *
 *   Phi = sum_i ( -M_i / r_i )    (G = 1, c = 1)
 *
 * Phi < 0 always (attractive well).  The Schwarzschild time-dilation factor
 * is approximated as sqrt(1 + 2*Phi), which equals sqrt(1 - RS/r) for a
 * single BH (RS = 2*M).  Summing potentials before taking the square root
 * handles multiple black holes correctly via superposition.
 */
__accelerated static inline float
bh_potential(__global struct scene *scene, const vec3_t *pos)
{
	__global struct object *obj;
	float phi = 0.0f;

	list_for_each_entry(obj, &scene->notmesh_objects, entry) {
		__global struct blackhole *bh;
		vec3_t to_bh;
		float r2;

		if (obj->type != BLACKHOLE_OBJECT)
			continue;

		bh    = container_of(obj, typeof(*bh), obj);
		to_bh = v3_sub(bh->center, *pos);
		r2    = v3_dot(to_bh, to_bh);
		if (r2 > EPSILON)
			phi -= bh->mass / sqrtf(r2);
	}
	return phi;
}

/*
 * apply_color_shift__linear_shift -- energy-conserving linear spectral shift.
 * NOT USED -- kept for reference.
 *
 * Linear cross-talk: blue -> green -> red -> (IR lost) on redshift,
 * red -> green -> blue -> (UV lost) on blueshift.  Physically motivated
 * but makes white light turn yellow on redshift rather than clearly red.
 */
__accelerated static inline void
apply_color_shift__linear_shift(vec3_t *color, float z, float strength)
{
	float s, rs, bs;
	vec3_t out;

	if (!color || strength <= 0.0f || z == 1.0f)
		return;

	s  = fmaxf(-1.0f, fminf(1.0f, (z - 1.0f) * strength));
	rs = fmaxf(0.0f, -s); /* redshift amount (non-zero when s < 0) */
	bs = fmaxf(0.0f,  s); /* blueshift amount (non-zero when s > 0) */

	out.x = color->x * (1.0f - rs - bs) + color->y * rs;
	out.y = color->y * (1.0f - rs - bs) + color->z * rs + color->x * bs;
	out.z = color->z * (1.0f - rs - bs) + color->y * bs;

	*color = v3_muls(out, z);
}

/*
 * apply_color_shift -- apply gravitational spectral shift to *color.
 *
 * z is the Schwarzschild frequency ratio z = f_obs / f_emit:
 *   z < 1: redshift  -- photon climbed out of a deeper potential than the
 *                       camera, losing energy (common case: camera far away)
 *   z > 1: blueshift -- camera is deeper in the well than the emitter
 *   z = 1: no shift
 *
 * strength controls artistic exaggeration (0=off, 1=physical, >1=exaggerated).
 *
 * Artistically tuned: instead of linearly shifting channels (see
 * apply_color_shift__linear_shift), we selectively attenuate the
 * higher-frequency channels to leave a distinct red/blue tint, then
 * scale overall brightness by z.
 */
__accelerated static inline void
apply_color_shift(vec3_t *color, float z, float strength)
{
	float t;
	vec3_t out;

	if (!color || strength <= 0.0f || z == 1.0f)
		return;

	out = *color;

	if (z < 1.0f) {
		/* REDSHIFT: Object is deeper in the well.
		 * Blue dies extremely fast, Green dies moderately fast.
		 * Red persists the longest. */
		t = fminf(1.0f, (1.0f - z) * strength);

		out.y *= (1.0f - t * 0.8f);  /* Suppress Green */
		out.z *= (1.0f - t * 0.95f); /* Suppress Blue heavily */

	} else {
		/* BLUESHIFT: Camera is deeper in the well.
		 * Red dies extremely fast, Green dies moderately fast.
		 * Blue persists the longest. */
		t = fminf(1.0f, (z - 1.0f) * strength);

		out.x *= (1.0f - t * 0.95f); /* Suppress Red heavily */
		out.y *= (1.0f - t * 0.8f);  /* Suppress Green */
	}

	/* Total energy still scales by z (dimming for redshift, brightening for blueshift) */
	*color = v3_muls(out, z);
}

/*
 * bh_compute_z -- Schwarzschild frequency ratio z = f_obs / f_emit.
 *
 * Uses the Newtonian potential approximation:
 *
 *   time-dilation factor at point p: sqrt(1 + 2 * Phi(p))
 *
 * where Phi = sum_i(-M_i/r_i) is the total Newtonian potential (G=c=1).
 * For a single BH this equals sqrt(1 - RS/r), matching the exact formula.
 * Superposition lets it handle multiple BHs correctly.
 *
 *   z = sqrt(1 + 2*phi_emit) / sqrt(1 + 2*phi_cam)
 *
 * phi_emit = 0 for sky rays (photon originates at infinity, no potential).
 * phi_emit = bh_potential(hit_point) for object rays (actual emission site).
 *
 *   z < 1: redshift  (object deeper in well than camera)
 *   z > 1: blueshift (camera deeper in well than object, or sky ray with
 *                     camera inside gravity well)
 */
__accelerated static inline float
bh_compute_z(float phi_emit, float phi_cam)
{
	float z_emit = sqrtf(fmaxf(0.0f, 1.0f + 2.0f * phi_emit));
	float z_cam  = sqrtf(fmaxf(0.0f, 1.0f + 2.0f * phi_cam));
	return (z_cam > EPSILON) ? z_emit / z_cam : 0.0f;
}

/*
 * blackhole_march -- integrate a ray through a gravitational field using
 * Euler steps, checking for event horizon absorption and object hits.
 *
 * Drop-in replacement for ray_trace() for primary rays when black holes
 * are present in the scene.
 *
 * Before the loop a ray-sphere intersection test checks whether the ray
 * path comes within escape_dist of any BH at all; if not, a plain
 * ray_trace() is done immediately (zero march cost).
 *
 * Algorithm per step:
 *   1. Event horizon: check dist(pos, center) < RS for each black hole.
 *      Absorbed rays return immediately (black).
 *   2. Object intersection: call ray_trace() for a straight micro-segment
 *      of length DT.  Accept the hit only if near <= DT (within this step).
 *   3. Gravitational bend: dir = normalize(dir + gravity(pos) * DT)
 *      Euler integration step -- bends the direction toward the BH.
 *   4. Advance position: pos += dir * DT
 *   5. Escape check: once the ray is farther than escape_dist from every BH
 *      gravity is negligible.  Uses per-BH distance so it fires correctly
 *      regardless of where the camera started.
 *
 * Colour shift (bh_z):
 *   phi_cam is computed once from scene->cam.pos before the loop -- it is
 *   the same for every ray from this camera position.
 *
 *   sky escape:    phi_emit = 0   -> z = 1/sqrt(1+2*phi_cam) >= 1 when
 *                                    camera is inside the gravity well
 *   object hit:    phi_emit = bh_potential(hit_point) -> z <= 1 when the
 *                                    object is deeper than the camera
 *   absorbed:      returns without setting *bh_z; caller detects absorption
 *                  via isect->hit_object->type == BLACKHOLE_OBJECT
 *
 * *orig and *dir are updated to the final march position and direction so
 * that the caller can compute hit_point = *orig + *dir * isect->near as
 * usual (same convention as ray_trace).
 *
 * Returns true  -- hit object or event horizon (isect populated)
 * Returns false -- escaped to sky (*dir updated for sky colour sampling)
 */
__accelerated static inline bool
blackhole_march(__global struct scene *scene,
		vec3_t *orig, vec3_t *dir,
		struct intersection *isect,
		__global struct octant_queue_entry *q_entries,
		uint32_t q_depth,
		float *bh_z,        /* out: Schwarzschild z ratio for colour shift */
		float *bh_strength) /* out: colorshift_strength from BH params */
{
	vec3_t pos = *orig;
	vec3_t d   = *dir;
	float  DT               = 0.05f;
	float  escape_sq        = 100.0f * 100.0f;
	float  colorshift_strength= 1.0f;
	float  phi_cam          = 0.0f; /* gravitational potential at camera */
	uint32_t max_steps      = 1000;
	uint32_t step;

	/*
	 * Read march parameters from the first black hole.  DT controls the
	 * accuracy/speed tradeoff -- halving DT doubles the work but halves
	 * the integration error.  max_steps caps the total computation.
	 */
	{
		__global struct object *obj;
		list_for_each_entry(obj, &scene->notmesh_objects, entry) {
			__global struct blackhole *bh;

			if (obj->type != BLACKHOLE_OBJECT)
				continue;

			bh         = container_of(obj, typeof(*bh), obj);
			DT                = bh->DT;
			max_steps         = bh->max_steps;
			escape_sq         = bh->escape_dist * bh->escape_dist;
			colorshift_strength = bh->colorshift_strength;
			break;
		}
	}

	/*
	 * Compute camera gravitational potential once.  phi_cam is the same
	 * for all rays from this camera position.  scene->cam.pos is the true
	 * observer location; *orig may be a jittered lens-disk point when
	 * defocus blur is active.  Must be computed before the pre-march
	 * optimisation so the early-exit path can also apply colour shift.
	 */
	phi_cam = bh_potential(scene, &scene->cam.pos);

	/*
	 * Pre-march optimisation: if the ray's path does not intersect the
	 * escape sphere of any black hole (closest approach > escape_dist),
	 * gravity is negligible for the entire path and we can skip the march
	 * entirely.  This avoids thousands of steps for the common case of
	 * rays that don't pass near the BH.
	 *
	 * Closest-approach formula: given unit direction d and offset L from
	 * the BH to the ray origin, the perpendicular distance is
	 *   d_perp = |L - d * dot(d, L)|
	 * If the BH is behind the ray (dot(d, L) < 0, i.e. BH is in the
	 * direction away from d), the closest point is the ray origin itself.
	 */
	{
		bool near_any = false;
		__global struct object *obj;

		list_for_each_entry(obj, &scene->notmesh_objects, entry) {
			__global struct blackhole *bh;
			vec3_t L, perp;
			float tc, d2;

			if (obj->type != BLACKHOLE_OBJECT)
				continue;

			bh = container_of(obj, typeof(*bh), obj);
			L  = v3_sub(bh->center, pos); /* pos == *orig here */
			tc = v3_dot(d, L);

			if (tc <= 0.0f) {
				/* BH is behind (or beside) the ray; closest point is origin */
				d2 = v3_dot(L, L);
			} else {
				perp = v3_sub(L, v3_muls(d, tc));
				d2   = v3_dot(perp, perp);
			}

			if (d2 < escape_sq) {
				near_any = true;
				break;
			}
		}
		if (!near_any) {
			/*
			 * Ray won't come near any BH -- skip the march and do a
			 * plain intersection test. We MUST still calculate colour shift
			 * because the camera might be deep in a gravity well looking away.
			 */
			bool hit = ray_trace(scene, orig, dir, isect, PRIMARY_RAY,
					     q_entries, q_depth);

			if (hit) {
				vec3_t hit_pt   = v3_add(*orig, v3_muls(*dir, isect->near));
				float  phi_emit = bh_potential(scene, &hit_pt);
				*bh_z           = bh_compute_z(phi_emit, phi_cam);
			} else {
				*bh_z           = bh_compute_z(0.0f, phi_cam); /* Sky */
			}
			*bh_strength = colorshift_strength;
			return hit;
		}
	}

	for (step = 0; step < max_steps; step++) {
		__global struct object *obj;

		/*
		 * Step 1 -- event horizon check.
		 *
		 * Event horizon (RS = 2*G*M, G=1): any ray inside this radius
		 * cannot escape -- return the black hole as the hit object so
		 * the renderer knows to output black.
		 */
		list_for_each_entry(obj, &scene->notmesh_objects, entry) {
			__global struct blackhole *bh;
			vec3_t delta;
			float r_sq;

			if (obj->type != BLACKHOLE_OBJECT)
				continue;

			bh    = container_of(obj, typeof(*bh), obj);
			delta = v3_sub(pos, bh->center);
			r_sq  = v3_dot(delta, delta);

			if (r_sq < bh->RS * bh->RS) {
				/* absorbed -- caller outputs black, no shift needed */
				isect->hit_object = obj;
				isect->near       = 0.0f;
				*orig = pos;
				*dir  = d;
				return true;
			}
		}

		/*
		 * Step 2 -- object intersection within this micro-segment.
		 *
		 * We cast a ray from the current march position along the
		 * current (already bent) direction.  We only accept hits
		 * within DT world units -- farther hits belong to a future
		 * step and the ray may have bent away from them by then.
		 *
		 * phi_emit is computed at the actual hit point, not at the
		 * periapsis: the surface is the true emission site.
		 */
		{
			struct intersection si;

			si.hit_object = NULL;
			si.near       = INFINITY;

			if (ray_trace(scene, &pos, &d, &si, PRIMARY_RAY,
				      q_entries, q_depth) && si.near <= DT) {
				vec3_t hit_pt   = v3_add(pos, v3_muls(d, si.near));
				float  phi_emit = bh_potential(scene, &hit_pt);
				*bh_z        = bh_compute_z(phi_emit, phi_cam);
				*bh_strength = colorshift_strength;
				*isect = si;
				*orig  = pos;
				*dir   = d;
				return true;
			}
		}

		/*
		 * Step 3 -- gravitational bend (RK2 midpoint integration).
		 *
		 * Second-order Runge-Kutta (midpoint method) for the coupled ODE:
		 *   dpos/dt = d,   dd/dt = acc(pos)
		 *
		 * Half-step: evaluate gravity at pos, kick direction to midpoint,
		 *   advance position to midpoint along that direction.
		 * Full step: re-evaluate gravity at midpoint position, apply the
		 *   midpoint acceleration over the full DT, advance pos along the
		 *   midpoint direction.
		 *
		 * RK2 is second-order: error ~ DT^3 per step, ~ DT^2 total.
		 * Photon orbits near the photon sphere stay stable much longer
		 * than with Euler at the same DT.  Cost: two gravity evaluations
		 * per step instead of one.
		 */
		{
			vec3_t acc1, acc2, d_mid, pos_mid;

			acc1    = blackhole_gravity(scene, &pos);
			d_mid   = v3_norm(v3_add(d, v3_muls(acc1, DT * 0.5f)));
			pos_mid = v3_add(pos, v3_muls(d_mid, DT * 0.5f));

			acc2 = blackhole_gravity(scene, &pos_mid);
			d    = v3_norm(v3_add(d, v3_muls(acc2, DT)));
			pos  = v3_add(pos, v3_muls(d_mid, DT));
		}

		/*
		 * Step 5 -- escape check.
		 *
		 * Once the ray has moved outside escape_dist from every BH,
		 * gravity is negligible and the ray will not return.  Checking
		 * per-BH distance (not total travel from origin) means this
		 * fires correctly regardless of where the camera started.
		 * Placed after the advance so it never triggers on the very
		 * first step for cameras outside the sphere (the pre-march
		 * ray-sphere test already handled that fast path above).
		 *
		 * phi_emit = 0 for sky: the photon originates at infinity,
		 * so z = 1/sqrt(1 + 2*phi_cam).
		 */
		{
			bool escaped = true;

			list_for_each_entry(obj, &scene->notmesh_objects, entry) {
				__global struct blackhole *bh;
				vec3_t to_bh;

				if (obj->type != BLACKHOLE_OBJECT)
					continue;

				bh    = container_of(obj, typeof(*bh), obj);
				to_bh = v3_sub(bh->center, pos);
				if (v3_dot(to_bh, to_bh) < escape_sq) {
					escaped = false;
					break;
				}
			}
			if (escaped) {
				*bh_z        = bh_compute_z(0.0f, phi_cam);
				*bh_strength = colorshift_strength;
				*dir = d;
				return false;
			}
		}
	}

	/* max_steps exhausted -- treat as escaped */
	*bh_z        = bh_compute_z(0.0f, phi_cam);
	*bh_strength = colorshift_strength;
	*dir = d;
	return false;
}

#endif /* BLACKHOLE_H */
