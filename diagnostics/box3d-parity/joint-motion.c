// SPDX-License-Identifier: MIT
#define NDEBUG
#include "box3d/box3d.h"
#include "joint.h"
#include "body.h"
#include "physics_world.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static void words(const void* p,int n) {printf("[");for(int i=0;i<n;++i){uint32_t u;memcpy(&u,(const char*)p+4*i,4);printf("%s%u",i?",":"",u);}printf("]");}
static void number(float x) {uint32_t u;memcpy(&u,&x,4);printf("%u",u);}
static void motion(b3WorldId w,int id) {
 b3World* world=b3GetWorld(w.index1-1);b3Body* b=b3Array_Get(world->bodies,id);b3BodySim* s=b3GetBodySim(world,b);b3BodyState* state=b3GetBodyState(world,b);
 b3Vec3 v=state?state->linearVelocity:b3Vec3_zero,a=state?state->angularVelocity:b3Vec3_zero;
 float f[]={s->transform.p.x,s->transform.p.y,s->transform.p.z,s->transform.q.v.x,s->transform.q.v.y,s->transform.q.v.z,s->transform.q.s,s->center.x,s->center.y,s->center.z,s->localCenter.x,s->localCenter.y,s->localCenter.z,v.x,v.y,v.z,a.x,a.y,a.z};words(f,19);
}
int main(void) {
 puts("[");int first=1;
 for(int kind=0;kind<3;++kind)for(int mode=0;mode<6;++mode)for(int sample=0;sample<8;++sample) {
  b3WorldDef wd=b3DefaultWorldDef();wd.gravity=b3Vec3_zero;b3WorldId w=b3CreateWorld(&wd);
  b3BodyDef d=b3DefaultBodyDef();d.type=mode==1?b3_staticBody:mode==3?b3_kinematicBody:b3_dynamicBody;d.position=(b3Pos){0.3f,-1.7f,2.1f};d.rotation=b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){1,2,-3}),sample*0.7f);b3BodyId a=b3CreateBody(w,&d);
  d.type=mode==2?b3_staticBody:mode==4?b3_kinematicBody:b3_dynamicBody;d.position=(b3Pos){-0.2f,0.4f,-1.3f};d.rotation=b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){-2,3,1}),sample*-0.31f);b3BodyId b=b3CreateBody(w,&d);
  b3ShapeDef sd=b3DefaultShapeDef();b3Sphere sphere={{0.3f,-0.2f,0.1f},0.7f};b3CreateSphereShape(a,&sd,&sphere);sphere.center=(b3Vec3){-0.4f,0.5f,-0.6f};b3CreateSphereShape(b,&sd,&sphere);
  b3Body_SetLinearVelocity(a,(b3Vec3){1.3f,-0.7f,0.2f});b3Body_SetAngularVelocity(a,(b3Vec3){-0.2f,0.6f,-1.1f});
  b3Body_SetLinearVelocity(b,(b3Vec3){-1.7f,0.4f,-0.3f});b3Body_SetAngularVelocity(b,(b3Vec3){0.3f,-0.8f,1.2f});
  b3JointId j;
  if(kind==0){b3PrismaticJointDef def=b3DefaultPrismaticJointDef();def.base.bodyIdA=a;def.base.bodyIdB=b;j=b3CreatePrismaticJoint(w,&def);}
  else if(kind==1){b3DistanceJointDef def=b3DefaultDistanceJointDef();def.base.bodyIdA=a;def.base.bodyIdB=b;j=b3CreateDistanceJoint(w,&def);}
  else {b3WheelJointDef def=b3DefaultWheelJointDef();def.base.bodyIdA=a;def.base.bodyIdB=b;j=b3CreateWheelJoint(w,&def);}
  b3Joint_SetLocalFrameA(j,(b3Transform){{-0.4f,0.2f,-0.1f},b3MakeQuatFromAxisAngle(b3Vec3_axisX,sample*0.11f)});
  b3Joint_SetLocalFrameB(j,(b3Transform){{0.7f,-0.3f,0.9f},b3MakeQuatFromAxisAngle(b3Vec3_axisY,sample*-0.13f)});
  if(mode==5){b3Body_SetAwake(a,false);b3Body_SetAwake(b,false);}
  b3JointSim* s=b3GetJointSimCheckType(j,b3Joint_GetType(j));
  // Emit only fields consumed by readers; unused native payload padding is not an input.
  uint32_t joint[111]={0};memcpy(joint,s,4*4);memcpy((char*)joint+offsetof(b3JointSim,localFrameA),&s->localFrameA,sizeof(b3Transform)*2);
  printf("%s{\"kind\":%d,\"mode\":%d,\"sample\":%d,\"a\":",first?"":",\n",s->type,mode,sample);first=0;motion(w,0);printf(",\"b\":");motion(w,1);printf(",\"joint\":");words(joint,111);printf(",\"output\":[");
  if(kind==0)number(b3PrismaticJoint_GetTranslation(j));else printf("null");printf(",");
  if(kind==1)number(b3DistanceJoint_GetCurrentLength(j));else printf("null");printf(",");
  if(kind==0)number(b3PrismaticJoint_GetSpeed(j));else printf("null");printf(",");
  if(kind==2)number(b3WheelJoint_GetSpinSpeed(j));else printf("null");printf("]}");
  b3DestroyWorld(w);
 }
 puts("\n]");
}
