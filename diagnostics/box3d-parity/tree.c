#include "box3d/box3d.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static b3DynamicTree* current;
static int mode;
static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static b3AABB box(int i) {
    float x = (i * 7) % 19;
    float y = i % 5, z = (i * 3) % 11, s = i % 3 + 1;
    return (b3AABB){{x,y,z},{x+s,y+s,z+s}};
}
static void report(b3DynamicTree* t) {
    b3DynamicTree_Validate(t);
    printf("tree %d %d %d %d %d %u\n", t->root, t->nodeCount, t->nodeCapacity, t->proxyCount, b3DynamicTree_GetHeight(t), bits(b3DynamicTree_GetAreaRatio(t)));
    for(int i=0;i<t->nodeCapacity;i++) {
        b3TreeNode* n=t->nodes+i;
        if (!(n->flags & b3_allocatedNode)) continue;
        bool leaf = (n->flags & b3_leafNode) != 0;
        printf("node %d %d %d %d %d %d %llu", i,n->parent,leaf?-1:n->children.child1,leaf?-1:n->children.child2,n->height,leaf, (unsigned long long)(leaf?n->userData:0));
        printf(" %u %u %u %u %u %u\n",bits(n->aabb.lowerBound.x),bits(n->aabb.lowerBound.y),bits(n->aabb.lowerBound.z),bits(n->aabb.upperBound.x),bits(n->aabb.upperBound.y),bits(n->aabb.upperBound.z));
    }
}
static bool overlap(int id,uint64_t user,void* context) {
    (void)context; printf("overlap %d %llu\n",id,(unsigned long long)user); return mode != 1;
}
static float ray(const b3RayCastInput* input,int id,uint64_t user,void* context) {
    (void)context; printf("ray %d %llu %u\n",id,(unsigned long long)user,bits(input->maxFraction));
    return mode == 1 ? 0 : mode == 2 ? -1 : mode == 3 ? 0.5f : 1;
}
static float closest(float limit,int id,uint64_t user,void* context) {
    (void)context; printf("closest %d %llu %u\n",id,(unsigned long long)user,bits(limit));
    b3Vec3 p={24,10,3}, q=b3ClosestPointToAABB(p,current->nodes[id].aabb);
    return b3DistanceSquared(p,q);
}
static float tiny(float limit,int id,uint64_t user,void* context) {
    (void)context; printf("tiny %d %llu %u\n", id, (unsigned long long)user, bits(limit)); return limit;
}
static void queries(b3DynamicTree* t) {
    current=t;
    float distance=0x1p-149f;
    b3TreeStats tinyStats=b3DynamicTree_QueryClosest(t,(b3Vec3){0,0,0},UINT64_MAX,false,tiny,NULL,&distance);
    printf("tiny-stats %d %d %u\n",tinyStats.nodeVisits,tinyStats.leafVisits,bits(distance));
    for(mode=0;mode<4;mode++) {
        uint64_t mask= mode==2 ? UINT64_C(0x100000000) : mode==3 ? UINT64_C(0x100000001) : UINT64_MAX;
        bool all=mode==3;
        b3TreeStats s=b3DynamicTree_Query(t,(b3AABB){{-2,-2,-2},{12,12,12}},mask,all,overlap,NULL);
        printf("stats %d %d\n",s.nodeVisits,s.leafVisits);
        b3RayCastInput r={{-5,2,4},{40,0,0},1};
        s=b3DynamicTree_RayCast(t,&r,mask,all,ray,NULL);
        printf("stats %d %d\n",s.nodeVisits,s.leafVisits);
        float distance=1000000;
        s=b3DynamicTree_QueryClosest(t,(b3Vec3){24,10,3},mask,all,closest,NULL,&distance);
        printf("stats %d %d %u\n",s.nodeVisits,s.leafVisits,bits(distance));
    }
}
int main(void) {
    b3DynamicTree t=b3DynamicTree_Create(16);
    report(&t); queries(&t);
    for(int i=0;i<48;i++) b3DynamicTree_CreateProxy(&t,box(i),UINT64_C(1)<<(i%3) | (i%2 ? UINT64_C(0x100000000):0),UINT64_C(0x1234567800000000)+i);
    report(&t); queries(&t);
    b3DynamicTree_Rebuild(&t,true); report(&t); queries(&t);
    b3DynamicTree_Rebuild(&t,false); report(&t); queries(&t);
    char path[256]; snprintf(path,sizeof(path),"/tmp/shallot-tree-%d.bin",(int)getpid());
    b3DynamicTree_Save(&t,path);
    b3DynamicTree loaded=b3DynamicTree_Load(path,2.0f); remove(path);
    report(&loaded); queries(&loaded);
    b3DynamicTree_Destroy(&loaded); b3DynamicTree_Destroy(&t);
    return 0;
}
