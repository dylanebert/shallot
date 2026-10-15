#include "box3d/math_functions.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float value(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static int count;
static void emit(b3Quat q) {
    printf("%s[%u,%u,%u,%u,%d]", count++ ? "," : "", bits(q.v.x), bits(q.v.y), bits(q.v.z), bits(q.s), b3IsValidQuat(q));
}
int main(void) {
    printf("[");
    b3Quat seeds[] = {
        {{0, 0, 0}, 1}, {{1, 0, 0}, 0}, {{1, 2, 3}, 4}, {{-4, 3, -2}, 1},
    };
    for (int seed = 0; seed < 4; ++seed) {
        b3Quat q = b3NormalizeQuat(seeds[seed]);
        for (int i = -64; i <= 64; ++i) {
            float scale = 1.0f + (float)i * 0x1p-24f;
            emit((b3Quat){{scale * q.v.x, scale * q.v.y, scale * q.v.z}, scale * q.s});
        }
    }
    emit((b3Quat){{0, 0, 0}, 0});
    uint32_t special[] = {0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x00000001, 0x7f7fffff};
    for (int i = 0; i < 6; ++i) {
        float x = value(special[i]);
        emit((b3Quat){{x, 0, 0}, 1});
        emit((b3Quat){{0, x, 0}, 1});
        emit((b3Quat){{0, 0, x}, 1});
        emit((b3Quat){{0, 0, 0}, x});
    }
    printf("]\n");
    return 0;
}
