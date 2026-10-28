#include "box3d/box3d.h"
#include <stdio.h>

static void vector(b3Vec3 v) { printf("[%.9g,%.9g,%.9g]", v.x, v.y, v.z); }
static void record(b3ContactData d) {
    printf("[");
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
    printf("]");
}
int main(void) {
    printf("[");
    for (int scene = 0; scene < 2; ++scene) {
        b3WorldDef wd = b3DefaultWorldDef();
        b3WorldId w = b3CreateWorld(&wd);
        b3BodyDef bd = b3DefaultBodyDef();
        b3BodyId ground = b3CreateBody(w, &bd);
        b3ShapeDef sd = b3DefaultShapeDef();
        b3MeshData* mesh = NULL;
        if (scene) {
            b3Vec3 vertices[] = {{-2,0.2f,-2},{-2,0.2f,2},{0,0,-2},{0,0,2},{2,0.2f,-2},{2,0.2f,2}};
            int32_t indices[] = {0,1,3,3,2,0,2,3,5,5,4,2};
            b3MeshDef md = {0};
            md.vertices = vertices; md.vertexCount = 6;
            md.indices = indices; md.triangleCount = 4; md.identifyEdges = true;
            mesh = b3CreateMesh(&md, NULL, 0);
            b3CreateMeshShape(ground, &sd, mesh, b3Vec3_one);
        } else {
            b3BoxHull box = b3MakeBoxHull(4, 0.5f, 4);
            b3CreateHullShape(ground, &sd, &box.base);
        }
        bd.type = b3_dynamicBody;
        bd.position = (b3Pos){0, scene ? 0.5f : 1.0f, 0};
        b3BodyId body = b3CreateBody(w, &bd);
        b3BoxHull box = b3MakeBoxHull(0.75f, 0.5f, 0.75f);
        b3ShapeId shape = b3CreateHullShape(body, &sd, &box.base);
        if (scene) printf(",");
        printf("[");
        for (int tick = 0; tick < 10; ++tick) {
            b3World_Step(w, 1.0f / 60.0f, 4);
            b3ContactData data[16];
            int count = b3Shape_GetContactData(shape, data, 16);
            if (tick) printf(",");
            if (count) record(data[0]); else printf("null");
        }
        printf("]");
        b3DestroyWorld(w);
        if (mesh) b3DestroyMesh(mesh);
    }
    printf("]\n");
}
