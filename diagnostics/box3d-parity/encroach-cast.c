#include "box3d/collision.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(void) {
    b3BoxHull ground = b3MakeTransformedBoxHull(3, 0.25f, 3, (b3Transform){{0, -0.25f, 0}, b3Quat_identity});
    b3BoxHull box = b3MakeBoxHull(0.5f, 0.5f, 0.5f);
    const b3Vec3 translations[] = {{1, 0.1f, 0}, {1, 0, 0}, {1, -0.1f, 0}, {0, -0.001f, 0}, {0, -0.1f, 0}};
    for (int i = 0; i < 5; ++i) {
        b3ShapeCastPairInput input = {0};
        input.proxyA = (b3ShapeProxy){ground.boxPoints, 8, 0};
        input.proxyB = (b3ShapeProxy){box.boxPoints, 8, 0};
        input.transform = (b3Transform){{0, 0.504f, 0}, b3Quat_identity};
        input.translationB = translations[i]; input.maxFraction = i == 4 ? 0.001f : 1;
        input.canEncroach = true;
        b3CastOutput out = b3ShapeCast(&input);
        printf("%d %08x %08x %08x %08x %08x %08x %08x %d\n", out.hit, bits(out.fraction), bits(out.point.x), bits(out.point.y), bits(out.point.z), bits(out.normal.x), bits(out.normal.y), bits(out.normal.z), out.iterations);
    }
}
