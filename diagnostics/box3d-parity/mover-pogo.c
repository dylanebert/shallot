// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include "joint.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static int first_field;
static void field(size_t offset, float value) { printf("%s[%zu,%u]", first_field ? "" : ",", offset / 4, bits(value)); first_field=0; }
#define FIELD(kind, member, value) field(offsetof(b3JointSim, kind) + offsetof(__typeof__(((b3JointSim*)0)->kind), member), value)
static void vector(b3Vec3 v) { printf("%u,%u,%u", bits(v.x), bits(v.y), bits(v.z)); }
int main(void) {
    printf("[");
    for (int kind=9; kind<=10; ++kind) for(int mode=0; mode<3; ++mode) {
        if(kind!=9 || mode) printf(",");
        b3WorldDef wd=b3DefaultWorldDef(); wd.gravity=b3Vec3_zero; wd.enableSleep=false;
        b3WorldId world=b3CreateWorld(&wd);
        b3BodyDef bd=b3DefaultBodyDef(); b3BodyId a=b3CreateBody(world,&bd);
        bd.type=b3_dynamicBody; bd.position=(b3Pos){0, mode==1?2.0f:0.5f,0};
        bd.linearVelocity=(b3Vec3){4,mode==1?4.0f:-4.0f,-3};
        b3BodyId b=b3CreateBody(world,&bd);
        b3Sphere sphere={.radius=0.5f}; b3ShapeDef sd=b3DefaultShapeDef(); b3CreateSphereShape(b,&sd,&sphere);
        b3MoverJointDef md=b3DefaultMoverJointDef(); md.base.bodyIdA=a; md.base.bodyIdB=b;
        md.linearVelocity=(b3Vec3){-6,0,8}; md.maxForce=0.75f;
        b3PogoJointDef pd=b3DefaultPogoJointDef(); pd.base.bodyIdA=a; pd.base.bodyIdB=b;
        pd.restLength=1; pd.hertz=4; pd.dampingRatio=0.7f; pd.maxTensionForce=0.25f; pd.maxCompressionForce=0.75f;
        pd.impulse=mode==2?0.03125f:0; pd.velocity=mode==2?-0.875f:0;
        b3JointId joint=kind==9?b3CreateMoverJoint(world,&md):b3CreatePogoJoint(world,&pd);
        printf("{\"kind\":%d,\"mode\":%d,\"fields\":[",kind,mode); first_field=1;
        if(kind==9) {
            FIELD(moverJoint,linearVelocity.x,md.linearVelocity.x); FIELD(moverJoint,linearVelocity.y,md.linearVelocity.y); FIELD(moverJoint,linearVelocity.z,md.linearVelocity.z);
            FIELD(moverJoint,maxForce,md.maxForce);
        } else {
            FIELD(pogoJoint,normal.x,pd.normal.x); FIELD(pogoJoint,normal.y,pd.normal.y); FIELD(pogoJoint,normal.z,pd.normal.z);
            FIELD(pogoJoint,restLength,pd.restLength); FIELD(pogoJoint,hertz,pd.hertz); FIELD(pogoJoint,dampingRatio,pd.dampingRatio);
            FIELD(pogoJoint,maxTensionForce,pd.maxTensionForce); FIELD(pogoJoint,maxCompressionForce,pd.maxCompressionForce);
            FIELD(pogoJoint,impulse,pd.impulse); FIELD(pogoJoint,velocity,pd.velocity);
        }
        printf("],\"samples\":[");
        for(int i=0;i<20;++i) {
            if(i && mode==2) {
                if(kind==10) { pd.impulse=b3PogoJoint_GetImpulse(joint); pd.velocity=b3PogoJoint_GetCurrentSpeed(joint); }
                b3DestroyJoint(joint,true);
                joint=kind==9?b3CreateMoverJoint(world,&md):b3CreatePogoJoint(world,&pd);
            }
            b3World_Step(world,1.0f/60.0f,4);
            b3Pos p=b3Body_GetPosition(b); b3Vec3 v=b3Body_GetLinearVelocity(b); b3Vec3 force=b3Joint_GetConstraintForce(joint);
            if(i) printf(","); printf("["); vector((b3Vec3){p.x,p.y,p.z}); printf(",");vector(v);printf(",");vector(force);
            if(kind==10) printf(",%u,%u",bits(b3PogoJoint_GetImpulse(joint)),bits(b3PogoJoint_GetCurrentSpeed(joint)));
            printf("]");
        }
        printf("]}"); b3DestroyWorld(world);
    }
    puts("]");
}
