#include "scene.h"

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

	color = ray_cast_for_pixel(scene, &orig, ix, iy, scale, img_ratio);
	color_vec_to_rgba32(&color, &scene->framebuffer[i]);
}
