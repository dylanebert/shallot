// Native side of mesh-impact.oracle.ts. Each stdin line is one case of float32 bit patterns in hex:
//   toi v1 v2 v3 scale c1 c2      b3ShapeTimeOfImpact of a 0.25 box swept from c1 to c2 over a
//                                 static one-triangle mesh
//   world <0 mesh|1 compound> v1 v2 v3 scale
//                                 one 1/60 s step of that box at (0, 1.5, 0) moving at -60 m/s
//                                 over the triangle, as a mesh shape or a compound's mesh child
#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "shape.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float value(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint32_t bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static int words(uint32_t* w, int n)
{
    for (int i = 0; i < n; ++i)
        if (scanf("%x", &w[i]) != 1)
            return 0;
    return 1;
}

static b3MeshData* triangle(const uint32_t* w)
{
    b3Vec3 v[3];
    for (int k = 0; k < 3; ++k)
        v[k] = (b3Vec3){value(w[3 * k]), value(w[3 * k + 1]), value(w[3 * k + 2])};
    int indices[3] = {0, 1, 2};
    b3MeshDef def = {0};
    def.vertices = v;
    def.vertexCount = 3;
    def.indices = indices;
    def.triangleCount = 1;
    return b3CreateMesh(&def, NULL, 0);
}

int main(void)
{
    b3BoxHull box = b3MakeBoxHull(0.25f, 0.25f, 0.25f);
    char mode[8];
    while (scanf("%7s", mode) == 1)
    {
        uint32_t w[18];
        if (strcmp(mode, "toi") == 0)
        {
            if (!words(w, 18))
                return 1;
            b3MeshData* mesh = triangle(w);
            b3Shape a = {0};
            a.type = b3_meshShape;
            a.sensorIndex = B3_NULL_INDEX;
            a.mesh = (b3Mesh){mesh, {value(w[9]), value(w[10]), value(w[11])}};
            b3Shape b = {0};
            b.type = b3_hullShape;
            b.sensorIndex = B3_NULL_INDEX;
            b.hull = &box.base;
            b3Sweep sweepA = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, b3Quat_identity, b3Quat_identity};
            b3Sweep sweepB = {{0, 0, 0},
                              {value(w[12]), value(w[13]), value(w[14])},
                              {value(w[15]), value(w[16]), value(w[17])},
                              b3Quat_identity,
                              b3Quat_identity};
            b3TOIOutput o = b3ShapeTimeOfImpact(&a, &b, &sweepA, &sweepB, 1.0f);
            printf("%d %08x %08x %08x %08x %08x %08x %08x %08x %d\n", o.state, bits(o.fraction),
                   bits(o.distance), bits(o.point.x), bits(o.point.y), bits(o.point.z),
                   bits(o.normal.x), bits(o.normal.y), bits(o.normal.z), o.usedFallback);
            b3DestroyMesh(mesh);
            continue;
        }
        int kind;
        if (strcmp(mode, "world") != 0 || scanf("%d", &kind) != 1 || !words(w, 12))
            return 1;
        b3MeshData* mesh = triangle(w);
        b3Vec3 scale = {value(w[9]), value(w[10]), value(w[11])};
        b3WorldDef worldDef = b3DefaultWorldDef();
        worldDef.gravity = (b3Vec3){0, 0, 0};
        worldDef.workerCount = 1;
        b3WorldId world = b3CreateWorld(&worldDef);
        b3BodyDef groundDef = b3DefaultBodyDef();
        b3BodyId ground = b3CreateBody(world, &groundDef);
        b3ShapeDef shapeDef = b3DefaultShapeDef();
        b3CompoundData* compound = NULL;
        if (kind == 0)
            b3CreateMeshShape(ground, &shapeDef, mesh, scale);
        else
        {
            b3SurfaceMaterial material = b3DefaultSurfaceMaterial();
            b3CompoundMeshDef child = {mesh, b3Transform_identity, scale, &material, 1};
            b3CompoundDef def = {0};
            def.meshes = &child;
            def.meshCount = 1;
            compound = b3CreateCompound(&def);
            b3CreateBakedCompoundShape(ground, &shapeDef, compound);
        }
        b3BodyDef bodyDef = b3DefaultBodyDef();
        bodyDef.type = b3_dynamicBody;
        bodyDef.position = (b3Pos){0, 1.5f, 0};
        bodyDef.linearVelocity = (b3Vec3){0, -60, 0};
        b3BodyId body = b3CreateBody(world, &bodyDef);
        b3CreateHullShape(body, &shapeDef, &box.base);
        b3World_Step(world, 1.0f / 60.0f, 4);
        b3WorldTransform t = b3Body_GetTransform(body);
        b3Vec3 v = b3Body_GetLinearVelocity(body);
        printf("%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", bits(t.p.x), bits(t.p.y),
               bits(t.p.z), bits(t.q.v.x), bits(t.q.v.y), bits(t.q.v.z), bits(t.q.s), bits(v.x),
               bits(v.y), bits(v.z));
        b3DestroyWorld(world);
        if (compound)
            b3DestroyCompound(compound);
        b3DestroyMesh(mesh);
    }
    return 0;
}
