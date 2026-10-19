// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static void words(const b3HullData* hull) {
    printf("[");
    for(int i=0;i<hull->byteCount/4;i++){uint32_t value;memcpy(&value,(const char*)hull+4*i,4);printf("%s%u",i?",":"",value);}
    printf("]");
}
// Each source hull, through b3CreateTransformedHullShape at every scale with one placement, as b3Shape_GetHull reads it.
int main(void) {
    b3WorldDef worldDef=b3DefaultWorldDef();
    b3WorldId world=b3CreateWorld(&worldDef);
    b3BodyDef bodyDef=b3DefaultBodyDef();
    b3ShapeDef shapeDef=b3DefaultShapeDef();
    b3BoxHull box=b3MakeBoxHull(0.4f,0.6f,0.5f);
    b3HullData* cylinder=b3CreateCylinder(1,0.5f,0,15);
    b3Vec3 points[]={{0,0,0},{2,0,0},{0,3,0},{0,0,4}};
    b3HullData* tetra=b3CreateHull(points,4,4);
    const b3HullData* sources[]={&box.base,cylinder,tetra};
    b3Vec3 scales[]={{1,1,1},{-0.75f,1,1},{1.2f,1,-0.9f},{0.9f,0.9f,0.9f},{1.25f,0.75f,1.5f}};
    float h=0.70710678118654752f;
    b3Transform transform={{2,-3,4},{{0,0,h},h}};
    printf("[");
    for(int source=0;source<3;source++)for(int scale=0;scale<5;scale++){
        b3BodyId body=b3CreateBody(world,&bodyDef);
        b3ShapeId shape=b3CreateTransformedHullShape(body,&shapeDef,sources[source],transform,scales[scale]);
        if(source||scale)printf(",");
        words(b3Shape_GetHull(shape));
    }
    printf("]\n");b3DestroyHull(cylinder);b3DestroyHull(tetra);b3DestroyWorld(world);return 0;
}
