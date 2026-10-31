#include "box3d/box3d.h"
#include "dynamic_tree.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
static b3DynamicTree tree;
static uint32_t bits(float f){uint32_t v;memcpy(&v,&f,4);return v;}
static float read_float(void){uint32_t v;if(scanf("%x",&v)!=1)return 0;float f;memcpy(&f,&v,4);return f;}
static b3Vec3 read_vec(void){b3Vec3 v;v.x=read_float();v.y=read_float();v.z=read_float();return v;}
static b3AABB read_box(void){b3AABB a;a.lowerBound=read_vec();a.upperBound=read_vec();return a;}
static void snapshot(void){
    printf("{\"end\":%d,\"capacity\":%d,\"pairFree\":%d,\"count\":%d,\"proxyCapacity\":%d,\"proxyFree\":%d,\"ordered\":%s,\"nodes\":[",tree.nodeEnd,tree.nodeCapacity,tree.pairFreeList,tree.proxyCount,tree.proxyCapacity,tree.proxyFreeList,tree.dfsOrdered?"true":"false");
    for(int i=0;i<tree.nodeEnd;i++){const b3TreeNode* n=tree.nodes+i;printf("%s[",i?",":"");const float* f=&n->aabb.lowerBound.x;for(int j=0;j<6;j++)printf("%s%u",j?",":"",bits(f[j]));printf(",%u,%u,%d]",n->flagIndex,(uint32_t)n->height,tree.parents[i]);}
    printf("],\"proxies\":[");for(int i=0;i<tree.proxyCapacity;i++){const b3TreeProxy* p=tree.proxies+i;printf("%s[\"%llu\",\"%llu\",%d,%d]",i?",":"",(unsigned long long)p->userData,(unsigned long long)p->categoryBits,p->node,p->next);}printf("]}\n");
}
static int hits[8192],hitCount,shrinkClosest;static float shrink;
static bool overlap(int id,uint64_t user,void* context){(void)user;(void)context;hits[hitCount++]=id;return true;}
static float ray(const b3RayCastInput* input,int id,uint64_t user,void* context){(void)user;(void)context;hits[hitCount++]=id;return shrink<0?input->maxFraction:shrink;}
static float box(const b3BoxCastInput* input,int id,uint64_t user,void* context){(void)user;(void)context;hits[hitCount++]=id;return input->maxFraction;}
static float closest(float distance,int id,uint64_t user,void* context){(void)user;hits[hitCount++]=id;if(!shrinkClosest)return distance;b3Vec3 p=*(b3Vec3*)context;return b3DistanceSquared(p,b3ClosestPointToAABB(p,b3DynamicTree_GetAABB(&tree,id)));}
int main(void){
    int capacity;if(scanf("%d",&capacity)!=1)return 1;tree=b3DynamicTree_Create(capacity);int handles[8192],handleCount=0,op;
    while(scanf("%d",&op)==1){
        if(op==0){b3AABB a=read_box();unsigned long long category,user;if(scanf("%llu %llu",&category,&user)!=2)return 2;handles[handleCount++]=b3DynamicTree_CreateProxy(&tree,a,category,user);}
        else if(op==1||op==2){int handle;if(scanf("%d",&handle)!=1)return 3;b3AABB a=read_box();if(op==1)b3DynamicTree_MoveProxy(&tree,handles[handle],a);else b3DynamicTree_EnlargeProxy(&tree,handles[handle],a);}
        else if(op==3){int handle;if(scanf("%d",&handle)!=1)return 4;b3DynamicTree_DestroyProxy(&tree,handles[handle]);}
        else if(op==4){int full;if(scanf("%d",&full)!=1)return 5;b3DynamicTree_Rebuild(&tree,full);}
        else if(op==5){b3DynamicTree_Validate(&tree);snapshot();}
        else {
            hitCount=0;b3TreeStats stats;float distance=FLT_MAX;unsigned long long mask;int all;
            if(op==6){b3AABB a=read_box();if(scanf("%llu %d",&mask,&all)!=2)return 6;stats=b3DynamicTree_Query(&tree,a,mask,all,overlap,NULL);}
            else if(op==7){b3RayCastInput input;input.origin=read_vec();input.translation=read_vec();input.maxFraction=read_float();if(scanf("%llu %d",&mask,&all)!=2)return 7;shrink=read_float();stats=b3DynamicTree_RayCast(&tree,&input,mask,all,ray,NULL);}
            else if(op==8){b3BoxCastInput input;b3AABB a=read_box();input.box=a;input.translation=read_vec();input.maxFraction=read_float();if(scanf("%llu %d",&mask,&all)!=2)return 8;stats=b3DynamicTree_BoxCast(&tree,&input,mask,all,box,NULL);}
            else if(op==9){b3Vec3 p=read_vec();if(scanf("%llu %d %d",&mask,&all,&shrinkClosest)!=3)return 9;stats=b3DynamicTree_QueryClosest(&tree,p,mask,all,closest,&p,&distance);}
            else return 10;
            printf("{\"hits\":[");for(int i=0;i<hitCount;i++)printf("%s%d",i?",":"",hits[i]);printf("],\"nodeVisits\":%d,\"leafVisits\":%d",stats.nodeVisits,stats.leafVisits);if(op==9)printf(",\"minDistanceSqr\":\"0x%08x\"",bits(distance));printf("}\n");
        }
    }b3DynamicTree_Destroy(&tree);return 0;
}
