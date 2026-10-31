#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "dynamic_tree.h"
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
        compound->tree.nodeEnd = 2 * leaves;
        compound->tree.proxyCount = leaves; compound->tree.pairFreeList = -1;
        nodes[1] = b3MakeEmptyNode();
        for (int i = 0; i < depth; ++i) {
            int parent = i == 0 ? 0 : 2 * i + 1;
            int pair = 2 * i + 2;
            nodes[parent].aabb = (b3AABB){{-0.5f,-0.5f,-0.5f},{100.5f,0.5f,0.5f}};
            nodes[parent].flagIndex = (uint32_t)pair;
            nodes[parent].height = depth-i;
            nodes[pair].aabb = (b3AABB){{99.5f,-0.5f,-0.5f},{100.5f,0.5f,0.5f}};
            nodes[pair].flagIndex = B3_LEAF_NODE | (uint32_t)i;
            nodes[pair].shapeIndex = i;
            compound->tree.proxies[i].node = pair;
        }
        int last = 2 * depth + 1;
        nodes[last].aabb = (b3AABB){{-0.5f,-0.5f,-0.5f},{0.5f,0.5f,0.5f}};
        nodes[last].flagIndex = B3_LEAF_NODE | (uint32_t)depth;
        nodes[last].shapeIndex = depth;
        compound->tree.proxies[depth].node = last;
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
