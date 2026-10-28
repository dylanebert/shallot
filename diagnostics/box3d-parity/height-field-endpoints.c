// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
struct b3World;
extern struct b3World* b3GetWorldFromId(b3WorldId);
extern uint64_t b3HashWorldState(struct b3World*);
static uint32_t word(float value) { uint32_t result; memcpy(&result, &value, 4); return result; }
static void bits(float value) { printf(",%u", word(value)); }
int main(void) {
    const int sizes[][2] = {{10,10},{4,1},{1,4},{2,2}};
    for (int i=0; i<4; ++i) {
        int rows=sizes[i][0], columns=sizes[i][1];
        b3WorldDef wd=b3DefaultWorldDef();
        b3WorldId world=b3CreateWorld(&wd);
        b3BodyDef bd=b3DefaultBodyDef();
        b3BodyId body=b3CreateBody(world,&bd);
        b3ShapeDef sd=b3DefaultShapeDef();
        b3HeightFieldData* field=b3CreateGrid(rows,columns,(b3Vec3){2,i==0?0:1.5f,2},false);
        b3CreateHeightFieldShape(body,&sd,field);
        bd.type=b3_dynamicBody;
        bd.position=(b3Pos){columns-1,2,rows-1};
        b3BodyId dynamic=b3CreateBody(world,&bd);
        b3Sphere sphere={b3Vec3_zero,0.5f};
        sd.filter.categoryBits=2;
        b3CreateSphereShape(dynamic,&sd,&sphere);
        b3QueryFilter filter=b3DefaultQueryFilter();
        filter.maskBits=1;
        for (int step=0; step<600; ++step) {
            b3World_Step(world,1.0f/60.0f,4);
            b3RayResult ray=b3World_CastRayClosest(world,(b3Pos){columns-1,4,rows-1},(b3Vec3){0,-8,0},filter);
            printf("%016llx %d [%d",(unsigned long long)b3HashWorldState(b3GetWorldFromId(world)),b3World_GetCounters(world).shapeCount,ray.hit);
            if(ray.hit) {
                bits((float)ray.point.x);bits((float)ray.point.y);bits((float)ray.point.z);
                bits(ray.normal.x);bits(ray.normal.y);bits(ray.normal.z);bits(ray.fraction);
            }
            puts("]");
        }
        b3Pos p=b3Body_GetPosition(dynamic);
        b3Vec3 v=b3Body_GetLinearVelocity(dynamic);
        printf("sphere %u %u %d\n",word((float)p.y),word(v.y),b3Body_IsAwake(dynamic));
        b3DestroyWorld(world);
        b3DestroyHeightField(field);
    }
    return 0;
}
