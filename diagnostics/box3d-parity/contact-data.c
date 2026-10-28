#define NDEBUG
#include "box3d/box3d.h"
#include "physics_world.h"
#include "contact.h"
#include <stdio.h>

static int children[64];
static void vector(b3Vec3 v) { printf("[%.9g,%.9g,%.9g]", v.x, v.y, v.z); }
static void record(b3ContactData d, b3WorldId world) {
    b3Contact* contact = b3Array_Get(b3GetWorldFromId(world)->contacts, d.contactId.index1 - 1);
    int child = contact->childIndex;
    children[d.contactId.index1] = child;
    printf("{\"shapeA\":%d,\"shapeB\":%d,\"childIndex\":%d,\"manifolds\":[", d.shapeIdA.index1, d.shapeIdB.index1, child);
    for (int m = 0; m < d.manifoldCount; ++m) {
        if (m) printf(",");
        const b3Manifold* v = d.manifolds + m;
        printf("["); vector(v->normal);
        printf(",%.9g,", v->twistImpulse); vector(v->frictionImpulse);
        printf(","); vector(v->rollingImpulse); printf(",[");
        for (int p = 0; p < v->pointCount; ++p) {
            if (p) printf(",");
            const b3ManifoldPoint* q = v->points + p;
            printf("["); vector(q->anchorA); printf(","); vector(q->anchorB);
            printf(",%.9g,%.9g,%.9g,%.9g,%.9g,%u,%d,%s]", q->separation, q->baseSeparation,
                q->normalImpulse, q->totalNormalImpulse, q->normalVelocity,
                q->featureId, q->triangleIndex, q->persisted ? "true" : "false");
        }
        printf("]]");
    }
    printf("]}");
}
static void events(b3WorldId world) {
    b3ContactEvents e = b3World_GetContactEvents(world);
    printf(",\"begin\":[");
    for (int i = 0; i < e.beginCount; ++i) {
        b3ContactBeginTouchEvent* v = e.beginEvents + i;
        if (i) printf(",");
        printf("[%d,%d,%d]", v->shapeIdA.index1, v->shapeIdB.index1, children[v->contactId.index1]);
    }
    printf("],\"end\":[");
    for (int i = 0; i < e.endCount; ++i) {
        b3ContactEndTouchEvent* v = e.endEvents + i;
        if (i) printf(",");
        printf("[%d,%d,%d]", v->shapeIdA.index1, v->shapeIdB.index1, children[v->contactId.index1]);
    }
    printf("]}");
}
int main(void) {
    printf("[");
    for (int scene = 0; scene < 3; ++scene) {
        b3WorldDef wd = b3DefaultWorldDef();
        b3WorldId w = b3CreateWorld(&wd);
        b3BodyDef bd = b3DefaultBodyDef();
        b3BodyId ground = b3CreateBody(w, &bd);
        b3ShapeDef sd = b3DefaultShapeDef();
        b3MeshData* mesh = NULL;
        b3CompoundData* compound = NULL;
        if (scene == 1) {
            b3Vec3 vertices[] = {{-2,0.2f,-2},{-2,0.2f,2},{0,0,-2},{0,0,2},{2,0.2f,-2},{2,0.2f,2}};
            int32_t indices[] = {0,1,3,3,2,0,2,3,5,5,4,2};
            b3MeshDef md = {0};
            md.vertices = vertices; md.vertexCount = 6;
            md.indices = indices; md.triangleCount = 4; md.identifyEdges = true;
            mesh = b3CreateMesh(&md, NULL, 0);
            b3CreateMeshShape(ground, &sd, mesh, b3Vec3_one);
        } else if (scene == 2) {
            b3CompoundSphereDef spheres[] = {
                {.sphere = {{-1,0,0},0.5f}, .material = b3DefaultSurfaceMaterial()},
                {.sphere = {{1,0,0},0.5f}, .material = b3DefaultSurfaceMaterial()}
            };
            b3CompoundDef cd = {0}; cd.spheres = spheres; cd.sphereCount = 2;
            compound = b3CreateCompound(&cd);
            b3CreateBakedCompoundShape(ground, &sd, compound);
        } else {
            b3BoxHull box = b3MakeBoxHull(4, 0.5f, 4);
            b3CreateHullShape(ground, &sd, &box.base);
        }
        bd.type = b3_dynamicBody;
        bd.position = (b3Pos){0, 3, 0};
        b3BodyId body = b3CreateBody(w, &bd);
        b3BoxHull box = b3MakeBoxHull(scene == 2 ? 1.75f : 0.75f, 0.5f, 0.75f);
        sd.enableContactEvents = true;
        b3ShapeId shape = b3CreateHullShape(body, &sd, &box.base);
        if (scene) printf(",");
        printf("[");
        for (int tick = 0; tick < 12; ++tick) {
            if (tick == 1) b3Body_SetTransform(body, (b3Pos){0, scene == 1 ? 0.5f : 1.0f, 0}, b3Quat_identity);
            if (tick == 11) b3Body_SetTransform(body, (b3Pos){0,20,0}, b3Quat_identity);
            b3World_Step(w, 1.0f / 60.0f, 4);
            b3ContactData data[16];
            int count = b3Shape_GetContactData(shape, data, 16);
            if (tick) printf(",");
            printf("{\"contacts\":[");
            for (int i = 0; i < count; ++i) { if (i) printf(","); record(data[i], w); }
            printf("]"); events(w);
        }
        printf("]");
        b3DestroyWorld(w);
        if (mesh) b3DestroyMesh(mesh);
        if (compound) b3DestroyCompound(compound);
    }
    printf("]\n");
}
