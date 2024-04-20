void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

typedef struct {
    float* data;
    float* gradient;
} parameter;

int main(int argc, char** argv) {
    int config[5] = {3, 5, 10, 5, 2};

    float* w1 = (float*)mallocCheck(sizeof(float) * config[0] * config[1]);  // (5, 3)
    float* w2 = (float*)mallocCheck(sizeof(float) * config[1] * config[2]);  // (10, 5)
    float* w3 = (float*)mallocCheck(sizeof(float) * config[2] * config[3]);  // (5, 10)
    float* w4 = (float*)mallocCheck(sizeof(float) * config[3] * config[4]);  // (2, 5)
    float* b1 = (float*)mallocCheck(sizeof(float) * config[1]);  // (5, 1)
    float* b2 = (float*)mallocCheck(sizeof(float) * config[2]);  // (10, 1)
    float* b3 = (float*)mallocCheck(sizeof(float) * config[3]);  // (5, 1)
    float* b4 = (float*)mallocCheck(sizeof(float) * config[4]);  // (2, 1)

    return 0;
}
