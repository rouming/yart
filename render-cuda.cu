/*
 * CUDA kernel for YART ray tracer.
 * Mirrors render-opencl.cl but compiled with nvcc.
 *
 * 'new' and 'this' are C++ keywords - remap them so the C headers compile cleanly.
 */
#define new new_val
#define this this_node
#include "ray-trace.h"
#undef new
#undef this

__global__ void render_cuda_kernel(struct scene *scene)
{
	float scale, img_ratio;
	vec3_t orig, color;
	uint32_t i, ix, iy;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= scene->width * scene->height)
		return;

	scale = tan(deg2rad(scene->fov * 0.5f));
	img_ratio = scene->width / (float)scene->height;

	orig = m4_mul_pos(scene->c2w, vec3(0.f, 0.f, 0.f));

	iy = i / scene->width;
	ix = i % scene->width;

	color = ray_cast_for_pixel(scene, &orig, ix, iy, scale, img_ratio);
	color_vec_to_rgba32(&color, &scene->framebuffer[i]);
}

extern "C" void cuda_invoke(struct scene *scene)
{
	int n = scene->width * scene->height;
	int block_size = 256;
	int num_blocks = (n + block_size - 1) / block_size;

	render_cuda_kernel<<<num_blocks, block_size>>>(scene);
	cudaDeviceSynchronize();
}
