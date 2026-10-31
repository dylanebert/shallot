#include "box3d/box3d.h"
#include <stdio.h>
#include <string.h>

static int count;
static void text(b3Pos p, const char* s, b3HexColor color, void* context)
{
    (void)color; (void)context;
    if (strncmp(s, "f = ", 4)) return;
    float force, torque;
    if (sscanf(s, "f = %f, t = %f", &force, &torque) != 2) return;
    printf("%s[{\"x\":%.17g,\"y\":%.17g,\"z\":%.17g},\"f = %.4g, t = %.4g\"]", count++ ? "," : "", p.x, p.y, p.z, force, torque);
}

int main(void)
{
    b3WorldDef wd = b3DefaultWorldDef();
    b3WorldId world = b3CreateWorld(&wd);
    for (int i = 0; i < 7; ++i)
    {
        b3BodyDef bd = b3DefaultBodyDef();
        bd.position = (b3Pos){i * 5 - 15, 0, 0};
        b3BodyId anchor = b3CreateBody(world, &bd);
        bd.type = b3_dynamicBody;
        bd.position = (b3Pos){i * 5 - 14, 2, 0};
        bd.rotation = b3MakeQuatFromAxisAngle((b3Vec3){0, 1, 0}, 0.4f);
        bd.angularVelocity = (b3Vec3){1, 2, 3};
        b3BodyId bob = b3CreateBody(world, &bd);
        b3ShapeDef sd = b3DefaultShapeDef();
        sd.density = 2;
        b3Sphere sphere = {{0.2f, 0, 0}, 0.5f};
        b3CreateSphereShape(bob, &sd, &sphere);
#define JOINT(index, Name) if(i == index){b3##Name##JointDef d=b3Default##Name##JointDef();d.base.bodyIdA=anchor;d.base.bodyIdB=bob;b3Create##Name##Joint(world,&d);}
        JOINT(0, Distance)
        JOINT(1, Parallel)
        JOINT(2, Prismatic)
        JOINT(3, Revolute)
        JOINT(4, Spherical)
        JOINT(5, Weld)
        JOINT(6, Wheel)
#undef JOINT
    }
    for(int i=0;i<4;++i)b3World_Step(world,1.0f/60.0f,4);
    b3DebugDraw draw=b3DefaultDebugDraw();
    draw.drawShapes=false;
    draw.drawJoints=draw.drawJointExtras=true;
    draw.DrawStringFcn=text;
    printf("[");
    b3World_Draw(world,&draw,UINT64_MAX);
    printf("]\n");
    b3DestroyWorld(world);
    return 0;
}
