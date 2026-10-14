// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include "joint.h"
#include "physics_world.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static b3JointId create(int kind, b3WorldId w, b3BodyId a, b3BodyId b) {
#define JOINT(kind, Name) case b3_##kind##Joint: { b3##Name##JointDef d=b3Default##Name##JointDef(); d.base.bodyIdA=a;d.base.bodyIdB=b;return b3Create##Name##Joint(w,&d); }
 switch(kind) {
 JOINT(parallel,Parallel) JOINT(distance,Distance) JOINT(filter,Filter) JOINT(motor,Motor)
 JOINT(prismatic,Prismatic) JOINT(revolute,Revolute) JOINT(spherical,Spherical) JOINT(weld,Weld) JOINT(wheel,Wheel)
 }
#undef JOINT
 return b3_nullJointId;
}
static void words(const void* p,int n) {printf("[");for(int i=0;i<n;++i){uint32_t u;memcpy(&u,(const char*)p+4*i,4);printf("%s%u",i?",":"",u);}printf("]");}
static void number(float x) { uint32_t u;memcpy(&u,&x,4);printf("%u",u); }
int main(void) {
 b3WorldDef wd=b3DefaultWorldDef();wd.gravity=b3Vec3_zero;b3WorldId w=b3CreateWorld(&wd);
 b3BodyDef bd=b3DefaultBodyDef();bd.type=b3_dynamicBody;b3BodyId a=b3CreateBody(w,&bd),b=b3CreateBody(w,&bd);
 puts("[");int first=1;
 for(int kind=0;kind<9;++kind)for(int sample=0;sample<32;++sample) {
  b3JointId id=create(kind,w,a,b);
  b3JointSim* s=b3GetJointSimCheckType(id,(b3JointType)kind);int jointId=s->jointId;
  memset(s,0,sizeof *s);s->jointId=jointId;s->type=(b3JointType)kind;s->bodyIdA=0;s->bodyIdB=1;
  b3Quat qa=b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){1,2,-3}),sample*0.17f);
  b3Quat qb=b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){-2,3,1}),sample*0.31f);
  b3Transform pa={{0.3f,-1.7f,2.1f},qa},pb={{-0.2f,0.4f,-1.3f},qb};
  if(sample==0)pb=pa;
  b3Body_SetTransform(a,pa.p,pa.q);b3Body_SetTransform(b,pb.p,pb.q);
  s->localFrameA=(b3Transform){{-0.4f,0.2f,-0.1f},b3MakeQuatFromAxisAngle(b3Vec3_axisX,sample*0.11f)};
  s->localFrameB=(b3Transform){{0.7f,-0.3f,0.9f},b3MakeQuatFromAxisAngle(b3Vec3_axisY,sample*-0.13f)};
  switch(kind) {
   case b3_distanceJoint:s->distanceJoint.length=2;s->distanceJoint.minLength=(sample&4)?8:0.5f;s->distanceJoint.maxLength=(sample&8)?1:9;s->distanceJoint.enableSpring=sample&1;s->distanceJoint.enableLimit=sample&2;break;
   case b3_prismaticJoint:s->prismaticJoint.enableLimit=sample&1;s->prismaticJoint.lowerTranslation=(sample&2)?4:-8;s->prismaticJoint.upperTranslation=(sample&4)?-4:8;break;
   case b3_revoluteJoint:s->revoluteJoint.enableLimit=sample&1;s->revoluteJoint.lowerAngle=(sample&2)?1:-2;s->revoluteJoint.upperAngle=(sample&4)?-1:2;break;
   case b3_sphericalJoint:s->sphericalJoint.enableConeLimit=sample&1;s->sphericalJoint.enableTwistLimit=sample&2;s->sphericalJoint.coneAngle=(sample&4)?0.3f:1.5f;s->sphericalJoint.lowerTwistAngle=-0.2f;s->sphericalJoint.upperTwistAngle=0.4f;break;
   case b3_weldJoint:s->weldJoint.linearHertz=(sample&1)?2:0;s->weldJoint.angularHertz=(sample&2)?3:0;break;
   case b3_wheelJoint:s->wheelJoint.enableSuspensionLimit=sample&1;s->wheelJoint.lowerSuspensionLimit=(sample&2)?4:-8;s->wheelJoint.upperSuspensionLimit=(sample&4)?-4:8;break;
  }
  printf("%s{\"kind\":%d,\"sample\":%d,\"a\":",first?"":",\n",kind,sample);first=0;words(&pa,7);printf(",\"b\":");words(&pb,7);printf(",\"joint\":");words(s,sizeof(*s)/4);printf(",\"output\":[");
  if(kind==b3_revoluteJoint)number(b3RevoluteJoint_GetAngle(id));else printf("null");printf(",");
  if(kind==b3_sphericalJoint)number(b3SphericalJoint_GetConeAngle(id));else printf("null");printf(",");
  if(kind==b3_sphericalJoint)number(b3SphericalJoint_GetTwistAngle(id));else printf("null");printf(",");
  if(kind==b3_wheelJoint)number(b3WheelJoint_GetSteeringAngle(id));else printf("null");printf(",");
  number(b3Joint_GetAngularSeparation(id));printf(",");number(b3Joint_GetLinearSeparation(id));printf("]}");
  b3DestroyJoint(id,false);
 }
 puts("\n]");b3DestroyWorld(w);
}
