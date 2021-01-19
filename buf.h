#ifndef BUF_H
#define BUF_H

/* Host allocation API */

struct opencl;

enum {
	BUF_MAP_WRITE = 1<<0,
	BUF_MAP_READ  = 1<<1,
	BUF_ZERO      = 1<<2,
};

void *buf_allocate(struct opencl *opencl, size_t sz);
void buf_destroy(void *ptr);
int buf_map(void *ptr, uint32_t flags);
int buf_unmap(void *ptr);

#endif /* BUF_H */
