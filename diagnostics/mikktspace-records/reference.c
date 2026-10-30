#include "mikktspace.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t vertices, corners;
    float *position, *normal, *uv, *tangent;
    uint32_t *indices;
} Mesh;

static int faces(const SMikkTSpaceContext *ctx) { return ((Mesh *)ctx->m_pUserData)->corners / 3; }
static int vertices(const SMikkTSpaceContext *ctx, int face) { (void)ctx; (void)face; return 3; }
static void position(const SMikkTSpaceContext *ctx, float *out, int face, int corner) {
    Mesh *m = ctx->m_pUserData;
    memcpy(out, m->position + 3 * m->indices[face * 3 + corner], 12);
}
static void normal(const SMikkTSpaceContext *ctx, float *out, int face, int corner) {
    Mesh *m = ctx->m_pUserData;
    memcpy(out, m->normal + 3 * m->indices[face * 3 + corner], 12);
}
static void uv(const SMikkTSpaceContext *ctx, float *out, int face, int corner) {
    Mesh *m = ctx->m_pUserData;
    memcpy(out, m->uv + 2 * m->indices[face * 3 + corner], 8);
}
static void tangent(const SMikkTSpaceContext *ctx, const float *value, float sign, int face, int corner) {
    Mesh *m = ctx->m_pUserData;
    float *out = m->tangent + 4 * (face * 3 + corner);
    memcpy(out, value, 12);
    out[3] = -sign;
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    FILE *input = fopen(argv[1], "rb");
    if (!input) return 1;
    Mesh m = {0};
    if (fread(&m.vertices, 4, 1, input) != 1 || fread(&m.corners, 4, 1, input) != 1) return 1;
    m.position = malloc(12 * m.vertices);
    m.normal = malloc(12 * m.vertices);
    m.uv = malloc(8 * m.vertices);
    m.indices = malloc(4 * m.corners);
    m.tangent = calloc(m.corners, 16);
    if (!m.position || !m.normal || !m.uv || !m.indices || !m.tangent) return 1;
    if (fread(m.position, 12, m.vertices, input) != m.vertices ||
        fread(m.normal, 12, m.vertices, input) != m.vertices ||
        fread(m.uv, 8, m.vertices, input) != m.vertices ||
        fread(m.indices, 4, m.corners, input) != m.corners) return 1;
    fclose(input);
    SMikkTSpaceInterface interface = {faces, vertices, position, normal, uv, tangent, NULL};
    SMikkTSpaceContext context = {&interface, &m};
    if (!genTangSpaceDefault(&context)) return 2;
    FILE *output = fopen(argv[2], "wb");
    if (!output || fwrite(m.tangent, 16, m.corners, output) != m.corners) return 1;
    fclose(output);
    free(m.position); free(m.normal); free(m.uv); free(m.indices); free(m.tangent);
    return 0;
}
