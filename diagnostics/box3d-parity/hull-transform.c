// SPDX-License-Identifier: MIT
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
int main(void) {
    b3BoxHull box=b3MakeBoxHull(0.4f,0.6f,0.5f);
    b3HullData* cylinder=b3CreateCylinder(2,1,-0.5f,12);
    b3Vec3 points[]={{0,0,0},{2,0,0},{0,3,0},{0,0,4}};
    b3HullData* tetra=b3CreateHull(points,4,4);
    const b3HullData* sources[]={&box.base,cylinder,tetra};
    b3Vec3 scales[]={{1,1,1},{2,0.5f,3},{-2,0.5f,3},{-2,-3,0.5f},{0.001f,0,-0.001f}};
    b3Transform transform={{2,-3,4},b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){1,2,3}),0.7f)};
    printf("[");
    for(int source=0;source<3;source++)for(int scale=0;scale<5;scale++){
        b3HullData* result=b3CloneAndTransformHull(sources[source],transform,scales[scale]);
        if(source||scale)printf(",");
        printf("{\"source\":");words(sources[source]);printf(",\"args\":[");
        float args[]={transform.p.x,transform.p.y,transform.p.z,transform.q.v.x,transform.q.v.y,transform.q.v.z,transform.q.s,scales[scale].x,scales[scale].y,scales[scale].z};
        for(int i=0;i<10;i++)printf("%s%.17g",i?",":"",args[i]);
        printf("],\"result\":");words(result);printf("}");b3DestroyHull(result);
    }
    printf("]\n");b3DestroyHull(cylinder);b3DestroyHull(tetra);return 0;
}
