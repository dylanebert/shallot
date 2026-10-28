#include <box3d/box3d.h>
#include <stdio.h>
#include <stdint.h>

int main(void)
{
    b3WorldDef wd = b3DefaultWorldDef();
    b3WorldId world = b3CreateWorld(&wd);
    b3BodyId bodies[4];
    b3ShapeId shapes[4];
    for (int i = 0; i < 4; ++i)
    {
        b3BodyDef bd = b3DefaultBodyDef();
        bd.type = i == 0 ? b3_staticBody : b3_dynamicBody;
        bd.position = (b3Pos){i == 0 ? 0 : i - 2, 0, 0};
        bd.gravityScale = 0;
        bodies[i] = b3CreateBody(world, &bd);
        b3ShapeDef sd = b3DefaultShapeDef();
        sd.isSensor = i == 0;
        sd.enableSensorEvents = true;
        b3Sphere sphere = {{0, 0, 0}, i == 0 ? 3.0f : 0.25f};
        shapes[i] = b3CreateSphereShape(bodies[i], &sd, &sphere);
    }
    printf("[");
    for (int tick = 0; tick < 4; ++tick)
    {
        if (tick == 1) b3Body_SetTransform(bodies[2], (b3Pos){10, 0, 0}, b3Quat_identity);
        if (tick == 2) b3Body_SetTransform(bodies[2], (b3Pos){0, 0, 0}, b3Quat_identity);
        if (tick == 3) b3Body_SetTransform(bodies[1], (b3Pos){10, 0, 0}, b3Quat_identity);
        b3World_Step(world, 1.0f / 60.0f, 4);
        b3ShapeId visitors[8];
        int count = b3Shape_GetSensorData(shapes[0], visitors, 8);
        printf("%s[", tick ? "," : "");
        for (int j = 0; j < count; ++j)
        {
            int id = -1;
            for (int i = 0; i < 4; ++i) if (B3_ID_EQUALS(visitors[j], shapes[i])) id = i;
            printf("%s%d", j ? "," : "", id);
        }
        printf("]");
    }
    printf("]\n");
    b3DestroyWorld(world);
}
