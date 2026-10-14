// SPDX-License-Identifier: MIT
// Native b3HullData byte gold, including padding, topology, SoA tails and hash.
#include "box3d/collision.h"
#include "box3d/math_functions.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int first = 1;
static void emit(const char* name, int operation, const float* args, int argc,
                 const b3Vec3* points, int count, int budget, const b3HullData* hull) {
    printf("%s{\"name\":\"%s\",\"operation\":%d,\"args\":[", first ? "" : ",\n", name, operation);
    first = 0;
    for (int i=0;i<argc;++i) { uint32_t u; memcpy(&u,args+i,4); printf("%s%u",i ? "," : "",u); }
    printf("],\"budget\":%d,\"points\":[",budget);
    for(int i=0;i<count;++i) {
        uint32_t u[3];memcpy(u,points+i,12);
        printf("%s[%u,%u,%u]",i ? "," : "",u[0],u[1],u[2]);
    }
    printf("],\"words\":");
    if (!hull) {printf("null}");return;}
    printf("[");
    for(int i=0;i<hull->byteCount/4;++i) {uint32_t u;memcpy(&u,(const char*)hull+4*i,4);printf("%s%u",i ? "," : "",u);}
    printf("]}");
}
static void cloud(const char* name,const b3Vec3* p,int n,int budget) {
    b3HullData* h=b3CreateHull(p,n,budget);emit(name,0,NULL,0,p,n,budget,h);if(h)b3DestroyHull(h);
}
static void quiet(const char* message) { (void)message; }
int main(void) {
    b3SetLogFcn(quiet);
    puts("[");
    b3Vec3 cube[8]={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
    cloud("cube",cube,8,8);cloud("cube-reduced",cube,8,4);cloud("budget-floor",cube,8,1);
    b3Vec3 authoringCube[8]={{1,1,1},{-1,1,1},{-1,-1,1},{1,-1,1},{1,1,-1},{-1,1,-1},{-1,-1,-1},{1,-1,-1}};
    cloud("authoring-cube",authoringCube,8,8);
    cloud("empty",cube,0,8);cloud("three-points",cube,3,8);cloud("coplanar",cube,4,8);
    b3Vec3 line[4]={{0,0,0},{1,0,0},{2,0,0},{3,0,0}};cloud("collinear",line,4,8);
    b3Vec3 same[4]={{1,1,1},{1,1,1},{1,1,1},{1,1,1}};cloud("coincident",same,4,8);
    b3Vec3 tetra[4]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};cloud("tetrahedron",tetra,4,4);
    b3Vec3 redundant[25];for(int i=0;i<24;++i)redundant[i]=cube[i%8];redundant[24]=(b3Vec3){0,0,0};
    cloud("weld-and-interior",redundant,25,128);
    b3Vec3 near[16];for(int i=0;i<8;++i){near[i]=cube[i];near[i+8]=b3Add(cube[i],(b3Vec3){0.0000001f,0,0});}
    cloud("near-coplanar",near,16,128);
    b3Vec3 shifted[8];for(int i=0;i<8;++i)shifted[i]=b3Add(cube[i],(b3Vec3){10000,-20000,30000});
    cloud("shifted",shifted,8,8);
    b3Vec3 sphere[512];uint32_t seed=12345;
    for(int i=0;i<512;++i){
        float a[3];for(int j=0;j<3;++j){seed=1664525*seed+1013904223;a[j]=(float)(seed>>8)*(1.0f/16777216.0f)*2.0f-1.0f;}
        sphere[i]=b3Normalize((b3Vec3){a[0],a[1],a[2]});
    }
    cloud("sphere-budget-4",sphere,512,4);cloud("sphere-budget-16",sphere,512,16);
    cloud("sphere-budget-40",sphere,512,40);cloud("sphere-native-ceiling",sphere,512,1000);
    float cylinders[][4]={{2,1,0,8},{3,0.75f,0.25f,6},{1,0.2f,-2,3},{2,1,0,32}};
    for(int i=0;i<4;++i){float* a=cylinders[i];b3HullData* h=b3CreateCylinder(a[0],a[1],a[2],(int)a[3]);char name[32];snprintf(name,sizeof(name),"cylinder-%d",i);emit(name,1,a,4,NULL,0,0,h);if(h)b3DestroyHull(h);}
    float cones[][4]={{2,1,0.5f,8},{3,0.75f,0.25f,6},{2,1,0,64},{2,1,0.5f,32}};
    for(int i=0;i<4;++i){float* a=cones[i];b3HullData* h=b3CreateCone(a[0],a[1],a[2],(int)a[3]);char name[32];snprintf(name,sizeof(name),"cone-%d",i);emit(name,2,a,4,NULL,0,0,h);if(h)b3DestroyHull(h);}
    for(int i=0;i<2;++i){float r=i ? 0.3f : 1.0f;b3HullData* h=b3CreateRock(r);emit(i ? "rock-small" : "rock",3,&r,1,NULL,0,0,h);if(h)b3DestroyHull(h);}
    float a[10]={0.5f,1,2,0,0,0,0,0,0,1};
    b3BoxHull box=b3MakeBoxHull(a[0],a[1],a[2]);emit("box",4,a,3,NULL,0,0,&box.base);
    box=b3MakeCubeHull(a[0]);emit("cube-box",5,a,1,NULL,0,0,&box.base);
    a[3]=2;a[4]=-3;a[5]=4;box=b3MakeOffsetBoxHull(a[0],a[1],a[2],(b3Vec3){a[3],a[4],a[5]});emit("offset-box",6,a,6,NULL,0,0,&box.base);
    b3Quat q=b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){1,2,3}),0.7f);
    a[6]=q.v.x;a[7]=q.v.y;a[8]=q.v.z;a[9]=q.s;
    box=b3MakeTransformedBoxHull(a[0],a[1],a[2],(b3Transform){.p={a[3],a[4],a[5]},.q=q});emit("transformed-box",7,a,10,NULL,0,0,&box.base);
    a[0]=0;a[1]=-1;a[2]=0.0001f;box=b3MakeBoxHull(a[0],a[1],a[2]);emit("box-minimum",4,a,3,NULL,0,0,&box.base);
    puts("\n]");return 0;
}
