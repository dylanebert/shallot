// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
struct b3World;
extern struct b3World* b3GetWorldFromId(b3WorldId);
extern uint64_t b3HashWorldState(struct b3World*);
static void bits(float value) { uint32_t word; memcpy(&word, &value, 4); printf(",%u", word); }
int main(void) {
    const int sizes[][2] = {{10,10},{10,1},{1,10},{1,1}};
    for (int i=0; i<4; ++i) {
        b3WorldDef wd=b3DefaultWorldDef();
        b3WorldId world=b3CreateWorld(&wd);
        b3BodyDef bd=b3DefaultBodyDef();
        b3BodyId body=b3CreateBody(world,&bd);
        b3ShapeDef sd=b3DefaultShapeDef();
        b3HeightFieldData* field=b3CreateGrid(sizes[i][0],sizes[i][1],(b3Vec3){2,i==0?0:1.5f,2},false);
        b3CreateHeightFieldShape(body,&sd,field);
        for (int step=0; step<3; ++step) {
            b3World_Step(world,1.0f/60.0f,4);
            b3RayResult ray=b3World_CastRayClosest(world,(b3Pos){0,4,0},(b3Vec3){0,-8,0},b3DefaultQueryFilter());
            printf("%016llx %d [%d",(unsigned long long)b3HashWorldState(b3GetWorldFromId(world)),b3World_GetCounters(world).shapeCount,ray.hit);
            if(ray.hit) {
                bits((float)ray.point.x);bits((float)ray.point.y);bits((float)ray.point.z);
                bits(ray.normal.x);bits(ray.normal.y);bits(ray.normal.z);bits(ray.fraction);
            }
            puts("]");
        }
        b3DestroyWorld(world);b3DestroyHeightField(field);
    }
    return 0;
}
