#include "box3d/box3d.h"
#include "box3d/collision.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool collect(b3ShapeId shape, void* context) {
    (void)shape;
    ++*(int*)context;
    return true;
}
int main(void) {
    for (int leaves = 8; leaves <= 1024; leaves = leaves == 8 ? 1023 : leaves + 1) {
        int depth = leaves - 1;
        b3CompoundSphereDef* spheres = calloc((size_t)leaves, sizeof(*spheres));
        for (int i = 0; i < leaves; ++i) {
            spheres[i].sphere = (b3Sphere){{i == depth ? 0.0f : 100.0f, 0, 0}, 0.5f};
            spheres[i].material = b3DefaultSurfaceMaterial();
        }
        b3CompoundDef def = {0}; def.spheres = spheres; def.sphereCount = leaves;
        b3CompoundData* compound = b3CreateCompound(&def);
        if (!compound || compound->tree.nodeCapacity < 2 * depth + 1) return 2;
        b3TreeNode* nodes = compound->tree.nodes;
        memset(nodes, 0, (size_t)compound->tree.nodeCapacity * sizeof(*nodes));
        compound->tree.root = 0; compound->tree.nodeCount = 2 * depth + 1;
        compound->tree.proxyCount = leaves; compound->tree.freeList = -1;
        for (int i = 0; i < depth; ++i) {
            nodes[i].aabb = (b3AABB){{-0.5f,-0.5f,-0.5f},{100.5f,0.5f,0.5f}};
            nodes[i].categoryBits = UINT64_MAX;
            nodes[i].children = (b3TreeNodeChildren){depth + i, i + 1 < depth ? i + 1 : 2 * depth};
            nodes[i].parent = i - 1; nodes[i].height = (uint16_t)(depth - i); nodes[i].flags = b3_allocatedNode;
            nodes[depth + i].aabb = (b3AABB){{99.5f,-0.5f,-0.5f},{100.5f,0.5f,0.5f}};
            nodes[depth + i].categoryBits = UINT64_MAX;
            nodes[depth + i].userData = (uint64_t)i; nodes[depth + i].parent = i;
            nodes[depth + i].flags = b3_allocatedNode | b3_leafNode;
        }
        nodes[2 * depth].aabb = (b3AABB){{-0.5f,-0.5f,-0.5f},{0.5f,0.5f,0.5f}};
        nodes[2 * depth].categoryBits = UINT64_MAX; nodes[2 * depth].userData = (uint64_t)depth;
        nodes[2 * depth].parent = depth - 1; nodes[2 * depth].flags = b3_allocatedNode | b3_leafNode;
        b3WorldDef wd = b3DefaultWorldDef(); wd.gravity = b3Vec3_zero;
        b3WorldId world = b3CreateWorld(&wd);
        b3BodyDef bd = b3DefaultBodyDef(); b3BodyId body = b3CreateBody(world, &bd);
        b3ShapeDef sd = b3DefaultShapeDef(); b3CreateBakedCompoundShape(body, &sd, compound);
        b3Vec3 point = b3Vec3_zero; b3ShapeProxy proxy = {&point, 1, 0.25f};
        b3QueryFilter filter = b3DefaultQueryFilter(); int count = 0;
        b3World_OverlapShape(world, (b3Pos){0,0,0}, &proxy, filter, collect, &count);
        bool overlap = b3Body_OverlapShape(body, (b3Pos){0,0,0}, &proxy, filter, b3Transform_identity);
        printf("%d %d %d\n", leaves, count, overlap);
        b3DestroyWorld(world); b3DestroyCompound(compound); free(spheres);
    }
}
