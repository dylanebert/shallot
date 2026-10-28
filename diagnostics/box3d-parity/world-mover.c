#include "box3d/box3d.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static b3BodyPlaneResult planes[4];
static int count, capacity;
static bool collect(b3ShapeId shape, const b3PlaneResult* results, int length, void* context)
{
    (void)context;
    for (int i = 0; i < length && count < capacity; ++i)
    {
        planes[count].shapeId = shape;
        planes[count++].result = results[i];
    }
    return count < capacity;
}
static void scalar(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    printf(" %08x", bits);
}
int main(void)
{
    b3WorldDef def = b3DefaultWorldDef();
    b3WorldId world = b3CreateWorld(&def);
    b3BodyDef bodyDef = b3DefaultBodyDef();
    bodyDef.position = (b3Pos){10, 2, -3};
    b3BodyId body = b3CreateBody(world, &bodyDef);
    for (int i = 0; i < 3; ++i)
    {
        b3ShapeDef shapeDef = b3DefaultShapeDef();
        shapeDef.userData = (void*)(uintptr_t)(i + 1);
        b3Sphere sphere = {{-0.5f + 0.5f * i, 0, 0}, 1};
        b3CreateSphereShape(body, &shapeDef, &sphere);
    }
    b3Capsule mover = {{0.8f, -0.2f, 0}, {0.8f, 0.2f, 0}, 0.5f};
    const int capacities[] = {0, 1, 2, 4};
    for (int c = 0; c < 4; ++c)
    {
        count = 0; capacity = capacities[c];
        if (capacity) b3World_CollideMover(world, bodyDef.position, &mover, b3DefaultQueryFilter(), collect, NULL);
        printf("%d %d", capacity, count);
        for (int i = 0; i < count; ++i)
        {
            b3PlaneResult p = planes[i].result;
            printf(" %d", (int)(uintptr_t)b3Shape_GetUserData(planes[i].shapeId));
            scalar(p.plane.normal.x); scalar(p.plane.normal.y); scalar(p.plane.normal.z); scalar(p.plane.offset);
            scalar(p.point.x); scalar(p.point.y); scalar(p.point.z);
            printf(" %d %d %d", p.triangleIndex, p.childIndex, p.materialIndex);
        }
        printf("\n");
    }
    b3Vec3 vertices[] = {{-1,0,-1}, {-1,0,1}, {1,0,1}, {1,0,-1}};
    int32_t indices[] = {0,1,2, 2,3,0};
    uint8_t materialIndices[] = {0,1};
    b3MeshDef meshDef = {0};
    meshDef.vertices = vertices; meshDef.vertexCount = 4;
    meshDef.indices = indices; meshDef.triangleCount = 2;
    meshDef.materialIndices = materialIndices;
    b3MeshData* mesh = b3CreateMesh(&meshDef, NULL, 0);
    bodyDef.position = (b3Pos){0,0,0};
    b3BodyId ground = b3CreateBody(world, &bodyDef);
    b3ShapeDef shapeDef = b3DefaultShapeDef();
    b3SurfaceMaterial materials[] = {b3DefaultSurfaceMaterial(), b3DefaultSurfaceMaterial()};
    shapeDef.materials = materials; shapeDef.materialCount = 2;
    b3CreateMeshShape(ground, &shapeDef, mesh, (b3Vec3){1,1,1});
    count = 0; capacity = 4;
    b3Capsule meshMover = {{0,-0.2f,0},{0,0.2f,0},0.5f};
    b3World_CollideMover(world, (b3Pos){0,0.4f,0}, &meshMover, b3DefaultQueryFilter(), collect, NULL);
    printf("mesh %d", count);
    for (int i = 0; i < count; ++i) {
        b3PlaneResult p = planes[i].result;
        scalar(p.plane.normal.x); scalar(p.plane.normal.y); scalar(p.plane.normal.z); scalar(p.plane.offset);
        scalar(p.point.x); scalar(p.point.y); scalar(p.point.z);
        printf(" %d %d %d", p.triangleIndex, p.childIndex, p.materialIndex);
    }
    printf("\n");
    b3DestroyWorld(world);
    b3DestroyMesh(mesh);
}
