#include "box3d/box3d.h"
#include "dynamic_tree.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static uint32_t bits(float f) { uint32_t v; memcpy(&v,&f,4); return v; }
int main(void) {
    int count;
    if(scanf("%d",&count)!=1 || count<1) return 1;
    b3DynamicTree t=b3DynamicTree_Create(count);
    for(int i=0;i<count;i++) {
        uint32_t w[6]; for(int j=0;j<6;j++) if(scanf("%x",w+j)!=1) return 2;
        b3AABB a; memcpy(&a,w,sizeof(a));
        b3DynamicTree_CreateProxy(&t,a,UINT64_MAX,(uint64_t)i);
    }
    b3DynamicTree_Rebuild(&t,true); b3DynamicTree_Validate(&t);
    printf("{\"root\":0,\"nodeCount\":%d,\"nodeCapacity\":%d,\"nodes\":[",t.nodeEnd,t.nodeEnd);
    int written=0;
    for(int i=0;i<t.nodeEnd;i++) {
        const b3TreeNode* n=t.nodes+i; if(b3IsEmptyNode(n))continue;
        bool leaf=b3IsLeaf(n);
        printf("%s{\"index\":%d,\"leaf\":%s,\"height\":%d,\"flags\":%d,\"aabb\":{\"lowerBound\":[",written++?",":"",i,leaf?"true":"false",leaf?0:n->height,leaf?5:1);
        const float* f=&n->aabb.lowerBound.x;
        for(int j=0;j<3;j++)printf("%s\"0x%08x\"",j?",":"",bits(f[j]));
        printf("],\"upperBound\":[");for(int j=3;j<6;j++)printf("%s\"0x%08x\"",j>3?",":"",bits(f[j]));
        printf("]}");
        if(leaf)printf(",\"userData\":\"%llu\"",(unsigned long long)t.proxies[b3GetProxyId(n)].userData);
        else printf(",\"child1\":%d,\"child2\":%d",b3GetLeftChild(n),b3GetLeftChild(n)+1);
        printf("}");
    }
    printf("]}\n");b3DynamicTree_Destroy(&t);return 0;
}
