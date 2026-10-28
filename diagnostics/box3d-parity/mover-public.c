#define NDEBUG
#include "box3d/box3d.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void scalar(float value) { uint32_t bits; memcpy(&bits, &value, 4); printf(" %08x", bits); }
static bool accept(b3ShapeId id, void* context) { (void)id; return *(bool*)context; }
static void inverse(b3BodyId body, int c, int step) {
    printf("inverse %d %d", c, step); scalar(b3Body_GetInverseMass(body));
    b3Matrix3 m = b3Body_GetWorldInverseRotationalInertia(body);
    scalar(m.cx.x); scalar(m.cx.y); scalar(m.cx.z); scalar(m.cy.x); scalar(m.cy.y); scalar(m.cy.z); scalar(m.cz.x); scalar(m.cz.y); scalar(m.cz.z); printf("\n");
}
int main(void) {
    for (int c = 0; c < 6; c++) {
        b3WorldDef wd = b3DefaultWorldDef(); b3WorldId world = b3CreateWorld(&wd);
        b3BodyDef bd = b3DefaultBodyDef(); bd.type = c == 0 ? b3_staticBody : c == 1 ? b3_kinematicBody : b3_dynamicBody;
        bd.position = (b3Pos){1,3,2}; bd.rotation = (b3Quat){{0,0.6f,0},0.8f}; bd.angularVelocity = (b3Vec3){1,-1,2};
        if(c==3)bd.motionLocks.angularX=bd.motionLocks.angularY=bd.motionLocks.angularZ=true; bd.isAwake = c != 4; bd.isEnabled = c != 5;
        b3BodyId body = b3CreateBody(world, &bd); b3ShapeDef sd = b3DefaultShapeDef(); b3BoxHull box = b3MakeBoxHull(0.5f,0.5f,1.0f); b3CreateHullShape(body,&sd,&box.base);
        inverse(body,c,0); b3World_Step(world,1.0f/60.0f,4); inverse(body,c,1); b3DestroyWorld(world);
    }
    b3WorldDef wd = b3DefaultWorldDef(); b3WorldId world = b3CreateWorld(&wd);
    b3BodyDef bd = b3DefaultBodyDef(); bd.position = (b3Pos){4,0,0}; b3BodyId body = b3CreateBody(world,&bd);
    b3ShapeDef sd = b3DefaultShapeDef(); b3Sphere sphere = {{0,0,0},1}; b3CreateSphereShape(body,&sd,&sphere);
    b3Capsule mover = {{0,-0.2f,0},{0,0.2f,0},0.25f};
    for(int c=0;c<7;c++) {
        b3Pos origin = b3Pos_zero; b3Vec3 delta = {8,0,0}; b3QueryFilter filter = b3DefaultQueryFilter(); bool include = c != 4;
        if(c==1) delta.x = -8; if(c==2) origin.x=4; if(c==3)filter.maskBits=0; if(c==6)origin.x=2.75f;
        printf("cast %d",c); scalar(b3World_CastMover(world,origin,&mover,delta,filter,c==4||c==5?accept:NULL,&include)); printf("\n");
    }
    b3DestroyWorld(world);
}
