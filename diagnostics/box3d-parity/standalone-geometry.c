// SPDX-License-Identifier: MIT
#include "box3d/collision.h"
#include "box3d/math_functions.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static void words(const void* data, int count) {
    printf("[");
    for (int i=0;i<count;i++) { uint32_t w; memcpy(&w,(const char*)data+4*i,4); printf("%s%u",i?",":"",w); }
    printf("]");
}
int main(void) {
    b3Vec3 scales[]={{1,1,1},{2,0.5f,3},{-2,0.5f,3},{-2,-3,0.5f},{0,0,0},{0.001f,-0.001f,2}};
    b3Transform t={{2,-3,4},b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){1,2,3}),0.7f)};
    printf("{\"transform\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],\"boxes\":[",t.p.x,t.p.y,t.p.z,t.q.v.x,t.q.v.y,t.q.v.z,t.q.s);
    for (int i=0;i<6;i++) {
        b3BoxHull box=b3MakeScaledBoxHull((b3Vec3){1,0.5f,0.25f},t,scales[i]);
        b3MassData m=b3ComputeHullMass(&box.base,2.3f);
        printf("%s{\"scale\":[%.17g,%.17g,%.17g],\"hull\":",i?",":"",scales[i].x,scales[i].y,scales[i].z);
        words(&box.base,box.base.byteCount/4); printf(",\"mass\":"); words(&m,13); printf("}");
    }
    printf("],\"capsules\":[");
    b3Capsule capsules[]={{{-1,0,0},{1,0,0},1},{{1,2,3},{-2,4,1},0.3f},{{2,3,4},{2,3,4},0.5f},{{0,0,0},{0,3,0},0}};
    for(int i=0;i<4;i++)for(int j=0;j<3;j++) {
        float density=j==0?0:j==1?1:2.3f;
        b3Capsule c=capsules[i]; b3MassData m=b3ComputeCapsuleMass(&c,density);
        printf("%s{\"input\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],\"mass\":",i||j?",":"",c.center1.x,c.center1.y,c.center1.z,c.center2.x,c.center2.y,c.center2.z,c.radius,density);
        words(&m,13);printf("}");
    }
    printf("]}\n");
}
