// SPDX-License-Identifier: MIT
#include "physics_world.h"
#include "box3d/box3d.h"
#include <stdio.h>
uint64_t b3HashWorldState( b3World* world );
static int comma = 0;
static void snapshot(b3WorldId world, b3BodyId body) {
    b3MassData m = b3Body_GetMassData(body);
    b3Vec3 v = b3Body_GetLinearVelocity(body);
    printf("%s{\"hash\":\"%016llx\",\"mass\":%.9g,\"center\":[%.9g,%.9g,%.9g],\"inertia\":[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g],\"velocity\":[%.9g,%.9g,%.9g]}", comma++ ? "," : "", (unsigned long long)b3HashWorldState(b3GetWorldFromId(world)), m.mass, m.center.x,m.center.y,m.center.z,m.inertia.cx.x,m.inertia.cx.y,m.inertia.cx.z,m.inertia.cy.x,m.inertia.cy.y,m.inertia.cy.z,m.inertia.cz.x,m.inertia.cz.y,m.inertia.cz.z,v.x,v.y,v.z);
}
int main(void) {
    printf("[");
    for(int c=0;c<6;++c) {
        b3WorldDef wd=b3DefaultWorldDef(); wd.workerCount=1;
        b3WorldId world=b3CreateWorld(&wd);
        b3BodyDef bd=b3DefaultBodyDef(); bd.type=c==0?b3_staticBody:c==1?b3_kinematicBody:b3_dynamicBody;
        bd.position=(b3Pos){1,3,2}; bd.angularVelocity=(b3Vec3){1,-1,2};
        if(c==3) bd.motionLocks.angularX=bd.motionLocks.angularY=bd.motionLocks.angularZ=true;
        if(c==4) bd.isAwake=false;
        if(c==5) bd.isEnabled=false;
        b3BodyId body=b3CreateBody(world,&bd);
        b3BoxHull box=b3MakeBoxHull(.5f,.5f,1);
        b3ShapeDef sd=b3DefaultShapeDef(); b3CreateHullShape(body,&sd,&box.base);
        b3World_Step(world,1.0f/60,4);
        snapshot(world,body);
        b3MassData m={.mass=c==1?0:3,.center={.5f,.5f,1},.inertia={{2,.25f,0},{.25f,3,0},{0,0,4}}};
        b3Body_SetMassData(body,m); snapshot(world,body);
        for(int i=0;i<3;++i) { b3World_Step(world,1.0f/60,4); snapshot(world,body); }
        m.mass=0; m.inertia=b3Mat3_zero; m.center=(b3Vec3){0,0,0};
        b3Body_SetMassData(body,m); snapshot(world,body);
        b3World_Step(world,1.0f/60,4); snapshot(world,body);
        b3DestroyWorld(world);
    }
    printf("]\n");
}
