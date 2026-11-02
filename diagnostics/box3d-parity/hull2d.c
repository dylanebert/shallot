// Box3D hull.c public 2D hull calls; Erin Catto's API and definitions, MIT.
#include "box3d/box3d.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int cases = 0;
static void points(const b3Point2D* p, int count) {
    printf("[");
    for (int i=0;i<count;i++) printf("%s[%.9g,%.9g,%.9g,%d]",i?",":"",p[i].p.x,p[i].p.y,p[i].separation,p[i].originalIndex);
    printf("]");
}
static void record(const b3Point2D* input, int count, int target) {
    b3Point2D scratch[1024], hull[2048], simplified[2048];
    memcpy(scratch,input,count*sizeof(b3Point2D));
    int hullCount=b3Hull2D(scratch,count,hull);
    memcpy(simplified,hull,hullCount*sizeof(b3Point2D));
    int simplifiedCount=b3SimplifyHull2D(simplified,hullCount,target);
    printf("%s[%d,",cases++?",\n":"",target);
    points(input,count);printf(",");points(hull,hullCount);printf(",");points(simplified,simplifiedCount);printf("]");
}
int main(void) {
    printf("[");
    b3Point2D input[1024]={0};
    record(input,0,3);
    input[0]=(b3Point2D){{-0.0f,0},-0.125f,INT32_MIN};record(input,1,3);
    input[1]=(b3Point2D){{0,0},0.25f,INT32_MAX};record(input,2,3);
    b3Vec2 sliver[]={{0,0},{-0.00305632968f,0.0000554900616f},{-0.00305632968f,0.0246563386f},{-0.00305632968f,0.0246562902f},{-0.00305632968f,0.0800221488f}};
    for(int i=0;i<5;i++) input[i]=(b3Point2D){sliver[i],0,i};
    record(input,5,3);record(input,5,5);
    input[0]=(b3Point2D){{0,0},1,4};input[1]=(b3Point2D){{0.0015f,0},2,3};input[2]=(b3Point2D){{0,1},3,2};input[3]=(b3Point2D){{1,1},4,1};input[4]=(b3Point2D){{1,0},5,0};
    record(input,5,4);
    for(int n=4;n<=65;n++) {
        int count=n==65?257:n;
        float radius=5.0f;
        float delta=2.0f*B3_PI/(float)count;
        for(int i=0;i<count;i++) { float angle=delta*(float)i;input[i]=(b3Point2D){{radius*cosf(angle),radius*sinf(angle)},(float)(i%7)*-0.03125f,i}; }
        record(input,count,3);record(input,count,4);record(input,count,6);record(input,count,count);
    }
    uint32_t seed=12345;
    for(int c=0;c<20;c++) {
        int count=4+13*c;
        for(int i=0;i<count;i++) {
            seed=seed*1664525u+1013904223u;float x=10.0f*((float)(seed>>8)/16777216.0f)-5.0f;
            seed=seed*1664525u+1013904223u;float y=10.0f*((float)(seed>>8)/16777216.0f)-5.0f;
            input[i]=(b3Point2D){{x,y},(float)(i%3)*0.01f,count-i};
        }
        record(input,count,4);record(input,count,6);
    }
    printf("]\n");
}
