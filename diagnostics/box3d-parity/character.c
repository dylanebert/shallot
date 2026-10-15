// SPDX-License-Identifier: MIT
// Movement phase from Box3D samples/mover.cpp at 47d7f7cc. Velocity replaces input policy.
#include "box3d/box3d.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    b3Vec3 position, velocity, up, center1, center2;
    float radius, dt, pogo, rest;
    uint32_t steep;
    double maxSlope;
    b3Vec3 center;
    b3Quat rotation;
    float maxSpeed;
} Input;
typedef struct {
    b3Vec3 position, velocity;
    float pogo;
    uint32_t groundState;
    b3Vec3 normal, groundVelocity;
    uint32_t count, planeCount, passes, iterations, impulseCount, bodies[8];
    b3Vec3 impulses[8], center;
    uint32_t groundShape, shapes[8];
} Output;
typedef struct { b3ShapeId shape; b3Plane plane; b3Pos point; } Contact;
typedef struct {
    uint32_t id, dynamic, awake;
    b3Vec3 center, linear, angular;
    float invMass;
    b3Matrix3 invInertia;
} Body;
_Static_assert(sizeof(Input) == 120, "input ABI");
_Static_assert(sizeof(Output) == 252, "output ABI");
_Static_assert(sizeof(Body) == 88, "body transcript");
static b3WorldId world;
static Contact contacts[8];
static int count;
static int first;
static uint32_t bits(float f) { uint32_t u; memcpy(&u,&f,4); return u; }
static void words(const void* p, int n) {
    const unsigned char* bytes = p;
    printf("[");
    for (int i=0;i<n;++i) { uint32_t u; memcpy(&u,bytes+4*i,4); printf("%s%u",i?",":"",u); }
    printf("]");
}
static void event(const char* kind, const void* args, int n, const void* result, int m) {
    printf("%s{\"kind\":\"%s\",\"args\":",first?"":",",kind); first=0;
    words(args,n); printf(",\"result\":"); words(result,m); printf("}");
}
static Body body(b3ShapeId shape) {
    b3BodyId id=b3Shape_GetBody(shape);
    Body b={0};
    b.id=id.index1-1; b.dynamic=b3Body_GetType(id)==b3_dynamicBody; b.awake=b3Body_IsAwake(id);
    b.center=b3Body_GetWorldCenter(id); b.linear=b3Body_GetLinearVelocity(id); b.angular=b3Body_GetAngularVelocity(id);
    b.invMass=b3Body_GetInverseMass(id); b.invInertia=b3Body_GetWorldInverseRotationalInertia(id);
    uint32_t arg=shape.index1-1;
    event("body",&arg,1,&b,22);
    return b;
}
static bool planes(b3ShapeId shape, const b3PlaneResult* results, int n, void* context) {
    b3Pos* origin=context;
    for (int i=0;i<n && count<8;++i) contacts[count++]=(Contact){shape,results[i].plane,b3OffsetPos(*origin,results[i].point)};
    return true;
}
static Output move(Input in) {
    Output out={0}; out.position=in.position; out.velocity=in.velocity; out.pogo=in.pogo; out.groundShape=UINT32_MAX;
    float rest=in.rest*in.radius, rayLength=rest+in.radius;
    b3Vec3 a=b3RotateVector(in.rotation,in.center1), b=b3RotateVector(in.rotation,in.center2);
    b3Pos origin=b3OffsetPos(in.position,a);
    b3Vec3 translation=b3MulSV(-rayLength,in.up);
    b3QueryFilter pogoFilter={.categoryBits=1,.maskBits=~2u};
    b3RayResult ray=b3World_CastRayClosest(world,origin,translation,pogoFilter);
    float rayArgs[6]={origin.x,origin.y,origin.z,translation.x,translation.y,translation.z};
    uint32_t rayResult[8]={0};
    int hit=ray.hit;
    if(hit) { rayResult[0]=ray.shapeId.index1-1; rayResult[1]=bits(ray.fraction); memcpy(rayResult+2,&ray.point,12); memcpy(rayResult+5,&ray.normal,12); }
    event("ray",rayArgs,6,rayResult,hit?8:0);
    if(hit) {
        out.groundShape=ray.shapeId.index1-1;
        Body ground=body(ray.shapeId);
        if(ground.awake) { out.center=ground.center; out.groundVelocity=b3Add(ground.linear,b3Cross(ground.angular,b3SubPos(ray.point,ground.center))); }
    }
    if(!hit || b3Dot(b3Sub(in.velocity,out.groundVelocity),in.up)>0.0f) { out.pogo=0.0f; out.groundVelocity=b3Vec3_zero; }
    else {
        out.normal=ray.normal; out.groundState=b3Dot(ray.normal,in.up)<cos(in.maxSlope)?2:1;
        float omega=2.0f*B3_PI*4.0f, omegaH=omega*in.dt;
        float length=ray.fraction*rayLength;
        out.pogo=(out.pogo-omega*omegaH*(length-rest))/(1.0f+2.0f*0.7f*omegaH+omegaH*omegaH);
    }
    b3Pos target=b3OffsetPos(b3OffsetPos(in.position,b3MulSV(in.dt,in.velocity)),b3MulSV(in.dt*out.pogo,in.up));
    b3CollisionPlane collision[8]={0};
    b3Capsule mover={a,b,in.radius};
    b3QueryFilter filter={.categoryBits=1,.maskBits=~0u,.id=1};
    b3QueryFilter castFilter={.categoryBits=1,.maskBits=~2u,.id=1};
    for(int iteration=0;iteration<5;++iteration) {
        count=0;
        b3World_CollideMover(world,out.position,&mover,filter,planes,&out.position);
        float args[10]={out.position.x,out.position.y,out.position.z,a.x,a.y,a.z,b.x,b.y,b.z,in.radius};
        uint32_t result[64]={0};
        for(int i=0;i<count;++i) {
            result[8*i]=contacts[i].shape.index1-1; memcpy(result+8*i+1,&contacts[i].plane,16); memcpy(result+8*i+5,&contacts[i].point,12);
            collision[i]=(b3CollisionPlane){.plane=contacts[i].plane,.pushLimit=FLT_MAX,.clipVelocity=true};
        }
        event("collide",args,10,result,count*8);
        b3PlaneSolverResult solved=b3SolvePlanes(b3SubPos(target,out.position),collision,count);
        out.passes++; out.iterations+=solved.iterationCount;
        b3Vec3 delta=solved.delta;
        float castArgs[6]={out.position.x,out.position.y,out.position.z,delta.x,delta.y,delta.z};
        float fraction=b3World_CastMover(world,out.position,&mover,delta,castFilter,NULL,NULL);
        event("cast",castArgs,6,&fraction,1);
        delta=b3MulSV(fraction,delta); out.position=b3OffsetPos(out.position,delta);
        if(b3LengthSquared(delta)<0.01f*0.01f) break;
    }
    for(int i=0;i<count;++i) {
        Body data=body(contacts[i].shape);
        if(!data.dynamic) continue;
        b3Vec3 normal=b3Neg(collision[i].plane.normal);
        out.center=data.center;
        b3Vec3 r=b3SubPos(contacts[i].point,data.center), rn=b3Cross(r,normal);
        float mass=data.invMass+b3Dot(rn,b3MulMV(data.invInertia,rn));
        float normalMass=mass>0.0f?1.0f/mass:0.0f;
        b3Vec3 vr=b3Add(data.linear,b3Cross(data.angular,r));
        float vn=b3Dot(b3Sub(vr,out.velocity),normal);
        b3Vec3 impulse=b3MulSV(b3MaxFloat(-normalMass*vn,0.0f),normal);
        out.velocity=b3MulSub(out.velocity,0.0f,impulse);
        uint32_t j=0; while(j<out.impulseCount && out.bodies[j]!=data.id) ++j;
        if(j==out.impulseCount) { out.bodies[j]=data.id; out.impulseCount++; }
        out.impulses[j]=b3Add(out.impulses[j],impulse);
        uint32_t args[7]={data.id}; memcpy(args+1,&impulse,12); memcpy(args+4,&contacts[i].point,12);
        event("impulse",args,7,NULL,0);
        b3Body_ApplyLinearImpulse(b3Shape_GetBody(contacts[i].shape),impulse,contacts[i].point,true);
    }
    out.velocity=b3ClipVector(out.velocity,collision,count);
    out.count=out.planeCount=count;
    for(int i=0;i<count;++i) out.shapes[i]=contacts[i].shape.index1-1;
    return out;
}
static b3BodyId box(b3Vec3 p, b3Vec3 half, b3Quat rotation, bool dynamic, bool awake) {
    b3BodyDef def=b3DefaultBodyDef(); def.position=p; def.rotation=rotation;
    def.type=dynamic?b3_dynamicBody:b3_staticBody; def.isAwake=awake;
    b3BodyId id=b3CreateBody(world,&def);
    b3BoxHull hull=b3MakeBoxHull(half.x,half.y,half.z);
    b3ShapeDef shape=b3DefaultShapeDef();
    b3CreateHullShape(id,&shape,&hull.base);
    return id;
}
int main(void) {
    const char* names[]={"free-60","free-fast","floor","wall","corner","low-step","high-step","slope","dynamic-awake","dynamic-sleep","rising","pogo-compressed","rotated-capsule","dynamic-double","dynamic-spin"};
    printf("[\n");
    for(int i=0;i<15;++i) {
        b3WorldDef def=b3DefaultWorldDef(); world=b3CreateWorld(&def);
        Input in={0}; in.rotation=b3Quat_identity; in.position=(b3Vec3){0,1.5f,0}; in.velocity=(b3Vec3){3,-8,1}; in.up=b3Vec3_axisY;
        in.center1=(b3Vec3){0,-0.5f,0}; in.center2=(b3Vec3){0,0.5f,0}; in.radius=0.3f; in.rest=3; in.dt=0.1f; in.maxSlope=(double)(B3_PI/4); in.maxSpeed=400;
        if(i<2) {in.position=b3Vec3_zero; in.velocity=(b3Vec3){i?1200:3,0,0}; in.dt=1.0f/60.0f;}
        if(i>=2) box((b3Vec3){0,-0.5f,0},(b3Vec3){10,0.5f,10},b3Quat_identity,false,false);
        if(i>=3 && i<=6) in.velocity.x=6;
        if(i==4) in.velocity.z=6;
        if(i==3 || i==4) box((b3Vec3){0.8f,1,0},(b3Vec3){0.1f,1,10},b3Quat_identity,false,false);
        if(i==4) box((b3Vec3){0,1,0.8f},(b3Vec3){10,1,0.1f},b3Quat_identity,false,false);
        if(i==5 || i==6) box((b3Vec3){0.8f, i==5?0.04f:0.15f,0},(b3Vec3){0.5f,i==5?0.04f:0.15f,2},b3Quat_identity,false,false);
        if(i==7) box((b3Vec3){0,0.1f,0},(b3Vec3){3,0.1f,3},b3MakeQuatFromAxisAngle(b3Vec3_axisZ,0.4f),false,false);
        if(i==8 || i==9) {in.position=(b3Vec3){0,1,0};in.velocity=(b3Vec3){6,0,0};box((b3Vec3){0.6f,1,0},(b3Vec3){0.3f,0.3f,0.3f},b3Quat_identity,true,i==8);}
        if(i==12) in.rotation=b3MakeQuatFromAxisAngle(b3Vec3_axisZ,0.7f);
        if(i==13 || i==14) {
            in.position=(b3Vec3){0,1,0}; in.velocity=(b3Vec3){6,0,0};
            b3BodyId id=box((b3Vec3){0.6f,1,0},(b3Vec3){0.3f,0.3f,0.3f},b3Quat_identity,true,true);
            if(i==13) { b3BoxHull hull=b3MakeBoxHull(0.3f,0.3f,0.3f); b3ShapeDef shape=b3DefaultShapeDef(); b3CreateHullShape(id,&shape,&hull.base); }
            else { b3Body_SetLinearVelocity(id,(b3Vec3){-2,0.1f,1}); b3Body_SetAngularVelocity(id,(b3Vec3){1,2,-3}); }
        }
        if(i==10) in.velocity=(b3Vec3){2,5,0};
        if(i==11) {in.position.y=1.0f;in.velocity=b3Vec3_zero;in.pogo=0.7f;}
        printf("%s{\"name\":\"%s\",\"input\":",i?",\n":"",names[i]); words(&in,30); printf(",\"events\":["); first=1;
        Output out=move(in);
        printf("],\"output\":");words(&out,63);printf("}");
        b3DestroyWorld(world);
    }
    printf("\n]\n");
}
