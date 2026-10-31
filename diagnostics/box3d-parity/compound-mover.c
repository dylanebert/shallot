#include "box3d/box3d.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static b3PlaneResult results[16];static int count;
static uint32_t bits(float f){uint32_t u;memcpy(&u,&f,4);return u;}
static float read_float(void){uint32_t u;if(scanf("%x",&u)!=1)return 0;float f;memcpy(&f,&u,4);return f;}
static b3Vec3 read_vec(void){b3Vec3 v;v.x=read_float();v.y=read_float();v.z=read_float();return v;}
static bool collect(b3ShapeId shape,const b3PlaneResult* p,int n,void* ctx){(void)shape;(void)ctx;for(int i=0;i<n&&count<16;i++)results[count++]=p[i];return true;}
static void vec(b3Vec3 v){printf("[\"0x%08x\",\"0x%08x\",\"0x%08x\"]",bits(v.x),bits(v.y),bits(v.z));}
int main(void){
    b3BoxHull box=b3MakeBoxHull(0.5f,0.5f,0.5f);b3CompoundHullDef hulls[2]={0};for(int i=0;i<2;i++){hulls[i].hull=&box.base;hulls[i].transform=b3Transform_identity;hulls[i].transform.p.x=i==0?-1:1;hulls[i].material=b3DefaultSurfaceMaterial();}
    b3CompoundDef compoundDef={0};compoundDef.hulls=hulls;compoundDef.hullCount=2;b3CompoundData* compound=b3CreateCompound(&compoundDef);
    b3WorldDef worldDef=b3DefaultWorldDef();b3WorldId world=b3CreateWorld(&worldDef);b3BodyDef bodyDef=b3DefaultBodyDef();b3BodyId body=b3CreateBody(world,&bodyDef);b3ShapeDef shapeDef=b3DefaultShapeDef();b3CreateBakedCompoundShape(body,&shapeDef,compound);
    int cases;if(scanf("%d",&cases)!=1)return 1;printf("[");for(int c=0;c<cases;c++){b3Capsule mover;mover.center1=read_vec();mover.center2=read_vec();mover.radius=read_float();count=0;b3World_CollideMover(world,(b3Pos){0},&mover,b3DefaultQueryFilter(),collect,NULL);printf("%s{\"count\":%d,\"planes\":[",c?",":"",count);for(int i=0;i<count;i++){b3PlaneResult p=results[i];printf("%s{\"normal\":",i?",":"");vec(p.plane.normal);printf(",\"offset\":\"0x%08x\",\"point\":",bits(p.plane.offset));vec(p.point);printf("}");}printf("]}");}printf("]\n");b3DestroyWorld(world);b3DestroyCompound(compound);return 0;
}
