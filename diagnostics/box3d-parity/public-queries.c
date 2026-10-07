#include "box3d/box3d.h"
#include "box3d/collision.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { SPHERE, CAPSULE, HULL, MESH, HEIGHTFIELD, COMPOUND };

typedef struct Fixture
{
    b3WorldId world;
    b3BodyId body;
    b3ShapeId shape;
    b3MeshData* mesh;
    b3HeightFieldData* height;
    b3CompoundData* compound;
} Fixture;

typedef struct Output
{
    int count;
} Output;

static float value(uint32_t bits)
{
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void word(uint32_t value)
{
    printf(" %08x", value);
}

static void scalar(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    word(bits);
}

static void vector(b3Vec3 value)
{
    scalar(value.x);
    scalar(value.y);
    scalar(value.z);
}

static void position(b3Pos value)
{
    scalar((float)value.x);
    scalar((float)value.y);
    scalar((float)value.z);
}

static uint32_t tag(b3ShapeId shape)
{
    return (uint32_t)(uintptr_t)b3Shape_GetUserData(shape);
}

static b3SurfaceMaterial material(uint64_t id)
{
    b3SurfaceMaterial result = b3DefaultSurfaceMaterial();
    result.userMaterialId = id;
    return result;
}

static Fixture make_fixture(int kind, bool dynamic, b3Pos position, b3Quat rotation)
{
    Fixture f = {0};
    b3WorldDef worldDef = b3DefaultWorldDef();
    worldDef.gravity = b3Vec3_zero;
    worldDef.enableSleep = false;
    f.world = b3CreateWorld(&worldDef);

    b3BodyDef bodyDef = b3DefaultBodyDef();
    bodyDef.type = dynamic ? b3_dynamicBody : b3_staticBody;
    bodyDef.position = position;
    bodyDef.rotation = rotation;
    f.body = b3CreateBody(f.world, &bodyDef);

    b3ShapeDef shapeDef = b3DefaultShapeDef();
    shapeDef.userData = (void*)(uintptr_t)(kind + 1);
    shapeDef.baseMaterial = material(0x100000000ull + (uint64_t)kind + 1);
    if (kind == SPHERE)
    {
        b3Sphere sphere = {{0.0f, -1.0f, 0.0f}, 1.0f};
        f.shape = b3CreateSphereShape(f.body, &shapeDef, &sphere);
    }
    else if (kind == CAPSULE)
    {
        b3Capsule capsule = {{0.0f, -1.65f, 0.0f}, {0.0f, -0.35f, 0.0f}, 0.35f};
        f.shape = b3CreateCapsuleShape(f.body, &shapeDef, &capsule);
    }
    else if (kind == HULL)
    {
        b3BoxHull box = b3MakeBoxHull(1.0f, 0.75f, 0.6f);
        f.shape = b3CreateHullShape(f.body, &shapeDef, &box.base);
    }
    else if (kind == MESH)
    {
        f.mesh = b3CreateBoxMesh((b3Vec3){0.0f, -0.75f, 0.0f}, (b3Vec3){1.0f, 0.75f, 0.6f}, true);
        b3SurfaceMaterial m = material(0x100000004ull);
        shapeDef.materials = &m;
        shapeDef.materialCount = 1;
        f.shape = b3CreateMeshShape(f.body, &shapeDef, f.mesh, (b3Vec3){1.0f, 1.0f, 1.0f});
    }
    else if (kind == HEIGHTFIELD)
    {
        f.height = b3CreateGrid(4, 4, (b3Vec3){1.0f, 1.0f, 1.0f}, false);
        b3SurfaceMaterial m = material(0x100000005ull);
        shapeDef.materials = &m;
        shapeDef.materialCount = 1;
        f.shape = b3CreateHeightFieldShape(f.body, &shapeDef, f.height);
    }
    else
    {
        b3BoxHull box = b3MakeBoxHull(1.0f, 0.75f, 0.6f);
        b3CompoundHullDef hull = {
            .hull = &box.base,
            .transform = {{0.0f, -0.75f, 0.0f}, b3Quat_identity},
            .material = material(0x100000006ull),
        };
        b3CompoundSphereDef spheres[2] = {
            {{{-0.625f, -0.2f, 0.0f}, 0.2f}, material(0x200000006ull)},
            {{{0.625f, -0.2f, 0.0f}, 0.2f}, material(0x300000006ull)},
        };
        b3CompoundDef compoundDef = {.hulls = &hull, .hullCount = 1, .spheres = spheres, .sphereCount = 2};
        f.compound = b3CreateCompound(&compoundDef);
        f.shape = b3CreateBakedCompoundShape(f.body, &shapeDef, f.compound);
    }
    return f;
}

static void destroy_fixture(Fixture* f)
{
    b3DestroyWorld(f->world);
    if (f->mesh != NULL) b3DestroyMesh(f->mesh);
    if (f->height != NULL) b3DestroyHeightField(f->height);
    if (f->compound != NULL) b3DestroyCompound(f->compound);
}

static bool overlap_callback(b3ShapeId shape, void* context)
{
    Output* out = context;
    word(tag(shape));
    out->count += 1;
    return true;
}

static float cast_callback(b3ShapeId shape, b3Pos point, b3Vec3 normal, float fraction,
                           uint64_t userMaterialId, int triangleIndex, int childIndex, void* context)
{
    Output* out = context;
    word(tag(shape));
    position(point);
    vector(normal);
    scalar(fraction);
    word((uint32_t)userMaterialId);
    word((uint32_t)(userMaterialId >> 32));
    word((uint32_t)triangleIndex);
    word((uint32_t)childIndex);
    out->count += 1;
    return fraction;
}

static bool mover_filter(b3ShapeId shape, void* context)
{
    Output* out = context;
    word(tag(shape));
    out->count += 1;
    return true;
}

static void emit_plane(const b3PlaneResult* plane)
{
    vector(plane->plane.normal);
    scalar(plane->plane.offset);
    vector(plane->point);
}

static bool plane_callback(b3ShapeId shape, const b3PlaneResult* planes, int planeCount, void* context)
{
    Output* out = context;
    word(tag(shape));
    word((uint32_t)planeCount);
    for (int i = 0; i < planeCount; ++i) emit_plane(planes + i);
    out->count += 1;
    return true;
}

static void emit_cast(bool hit, b3ShapeId shape, b3Pos point, b3Vec3 normal, float fraction,
                      uint64_t userMaterialId, int triangleIndex, int childIndex, bool includeChild)
{
    word((uint32_t)hit);
    if (!hit) return;
    word(tag(shape));
    position(point);
    vector(normal);
    scalar(fraction);
    word((uint32_t)userMaterialId);
    word((uint32_t)(userMaterialId >> 32));
    word((uint32_t)triangleIndex);
    if (includeChild) word((uint32_t)childIndex);
}

static void query_case(unsigned operation, int kind, const uint32_t* bits)
{
    b3Pos origin = {value(bits[0]), value(bits[1]), value(bits[2])};
    b3Vec3 translation = {value(bits[3]), value(bits[4]), value(bits[5])};
    float radius = value(bits[6]);
    float halfHeight = value(bits[7]);
    float moverRadius = value(bits[8]);
    float extent = value(bits[9]);
    float maxFraction = value(bits[10]);
    b3Pos fixturePosition = {value(bits[11]), value(bits[12]), value(bits[13])};
    b3Quat fixtureRotation = {{value(bits[14]), value(bits[15]), value(bits[16])}, value(bits[17])};
    Fixture f = make_fixture(kind, false, fixturePosition, fixtureRotation);
    b3Vec3 proxyPoint = b3Vec3_zero;
    b3ShapeProxy proxy = {&proxyPoint, 1, radius};
    b3Capsule mover = {{0.0f, -halfHeight, 0.0f}, {0.0f, halfHeight, 0.0f}, moverRadius};
    b3QueryFilter filter = b3DefaultQueryFilter();
    if (maxFraction < 0.0f)
    {
        filter.categoryBits = 0;
        filter.maskBits = 0;
        maxFraction = -maxFraction;
    }
    b3WorldTransform bodyTransform = {fixturePosition, fixtureRotation};
    Output output = {0};

    word(operation);
    word((uint32_t)kind);
    if (operation == 0)
    {
        b3Vec3 e = {extent, extent, extent};
        b3Vec3 c = {(float)origin.x, (float)origin.y, (float)origin.z};
        b3TreeStats stats = b3World_OverlapAABB(f.world, (b3AABB){b3Sub(c, e), b3Add(c, e)}, filter, overlap_callback, &output);
        word((uint32_t)output.count); word((uint32_t)stats.nodeVisits); word((uint32_t)stats.leafVisits);
    }
    else if (operation == 1)
    {
        b3TreeStats stats = b3World_OverlapShape(f.world, origin, &proxy, filter, overlap_callback, &output);
        word((uint32_t)output.count); word((uint32_t)stats.nodeVisits); word((uint32_t)stats.leafVisits);
    }
    else if (operation == 2)
    {
        b3TreeStats stats = b3World_CastRay(f.world, origin, translation, filter, cast_callback, &output);
        word((uint32_t)output.count); word((uint32_t)stats.nodeVisits); word((uint32_t)stats.leafVisits);
    }
    else if (operation == 3)
    {
        b3RayResult r = b3World_CastRayClosest(f.world, origin, translation, filter);
        emit_cast(r.hit, r.shapeId, r.point, r.normal, r.fraction, r.userMaterialId, r.triangleIndex, r.childIndex, true);
    }
    else if (operation == 4)
    {
        b3TreeStats stats = b3World_CastShape(f.world, origin, &proxy, translation, filter, cast_callback, &output);
        word((uint32_t)output.count); word((uint32_t)stats.nodeVisits); word((uint32_t)stats.leafVisits);
    }
    else if (operation == 5)
    {
        b3World_CollideMover(f.world, origin, &mover, filter, plane_callback, &output);
        word((uint32_t)output.count);
    }
    else if (operation == 6)
    {
        float fraction = b3World_CastMover(f.world, origin, &mover, translation, filter, mover_filter, &output);
        scalar(fraction); word((uint32_t)output.count);
    }
    else if (operation == 7)
    {
        b3BodyCastResult r = b3Body_CastRay(f.body, origin, translation, filter, maxFraction, bodyTransform);
        emit_cast(r.hit, r.shapeId, r.point, r.normal, r.fraction, r.userMaterialId, r.triangleIndex, 0, false);
    }
    else if (operation == 8)
    {
        b3BodyCastResult r = b3Body_CastShape(f.body, origin, &proxy, translation, filter, maxFraction, false, bodyTransform);
        emit_cast(r.hit, r.shapeId, r.point, r.normal, r.fraction, r.userMaterialId, r.triangleIndex, 0, false);
    }
    else if (operation == 9)
    {
        word((uint32_t)b3Body_OverlapShape(f.body, origin, &proxy, filter, bodyTransform));
    }
    else if (operation == 10)
    {
        b3Vec3 point;
        float distance = b3Body_GetClosestPoint(f.body, &point, (b3Vec3){(float)origin.x, (float)origin.y, (float)origin.z});
        vector(point); scalar(distance);
    }
    else if (operation == 11)
    {
        b3BodyPlaneResult planes[8];
        int count = b3Body_CollideMover(f.body, planes, 8, origin, &mover, filter, bodyTransform);
        word((uint32_t)count);
        for (int i = 0; i < count; ++i) { word(tag(planes[i].shapeId)); emit_plane(&planes[i].result); }
    }
    putchar('\n');
    destroy_fixture(&f);
}

static void emit_manifold(const b3Manifold* manifold)
{
    word((uint32_t)manifold->pointCount);
    vector(manifold->normal);
    scalar(manifold->twistImpulse);
    vector(manifold->frictionImpulse);
    vector(manifold->rollingImpulse);
    for (int i = 0; i < manifold->pointCount; ++i)
    {
        const b3ManifoldPoint* p = manifold->points + i;
        vector(p->anchorA); vector(p->anchorB);
        scalar(p->separation); scalar(p->baseSeparation); scalar(p->normalImpulse);
        scalar(p->totalNormalImpulse); scalar(p->normalVelocity);
        word(p->featureId); word((uint32_t)p->triangleIndex); word((uint32_t)p->persisted);
    }
}

static void contact_case(int kind, int convexKind, const uint32_t* bits)
{
    b3Pos fixturePosition = kind == HEIGHTFIELD ? (b3Pos){-1.5f, 0.0f, -1.5f} : (b3Pos){0.0f, 0.0f, 0.0f};
    Fixture f = make_fixture(kind, false, fixturePosition, b3Quat_identity);
    b3BodyDef bodyDef = b3DefaultBodyDef();
    bodyDef.type = b3_dynamicBody;
    bodyDef.position = (b3Pos){value(bits[0]), value(bits[1]), value(bits[2])};
    bodyDef.rotation = (b3Quat){{value(bits[3]), value(bits[4]), value(bits[5])}, value(bits[6])};
    b3BodyId body = b3CreateBody(f.world, &bodyDef);
    b3ShapeDef shapeDef = b3DefaultShapeDef();
    shapeDef.userData = (void*)(uintptr_t)99;
    shapeDef.enableContactEvents = true;
    if (convexKind == SPHERE)
    {
        b3Sphere sphere = {b3Vec3_zero, value(bits[7])};
        (void)b3CreateSphereShape(body, &shapeDef, &sphere);
    }
    else if (convexKind == CAPSULE)
    {
        b3Capsule capsule = {
            {0.0f, -value(bits[8]), 0.0f},
            {0.0f, value(bits[8]), 0.0f},
            value(bits[7]),
        };
        (void)b3CreateCapsuleShape(body, &shapeDef, &capsule);
    }
    else
    {
        b3BoxHull box = b3MakeBoxHull(value(bits[7]), value(bits[8]), value(bits[9]));
        (void)b3CreateHullShape(body, &shapeDef, &box.base);
    }
    b3World_Step(f.world, 1.0f / 60.0f, 4);
    b3ContactEvents events = b3World_GetContactEvents(f.world);
    printf("0000000c"); word((uint32_t)kind); word((uint32_t)convexKind); word((uint32_t)events.beginCount);
    for (int i = 0; i < events.beginCount; ++i)
    {
        b3ContactBeginTouchEvent* event = events.beginEvents + i;
        word(tag(event->shapeIdA)); word(tag(event->shapeIdB));
        word((uint32_t)b3Contact_IsValid(event->contactId));
        b3ContactData data = b3Contact_GetData(event->contactId);
        word(tag(data.shapeIdA)); word(tag(data.shapeIdB));
        word((uint32_t)data.manifoldCount);
        for (int j = 0; j < data.manifoldCount; ++j) emit_manifold(data.manifolds + j);
    }
    putchar('\n');
    destroy_fixture(&f);
}

int main(void)
{
    unsigned operation;
    int kind;
    uint32_t bits[18];
    while (scanf("%u %d", &operation, &kind) == 2)
    {
        if (kind < SPHERE || kind > COMPOUND) return 2;
        if (operation == 12)
        {
            int convexKind;
            if (scanf("%d", &convexKind) != 1) return 3;
            if (convexKind < SPHERE || convexKind > HULL) return 2;
            for (int i = 0; i < 10; ++i) if (scanf("%x", bits + i) != 1) return 3;
            contact_case(kind, convexKind, bits);
            continue;
        }
        for (int i = 0; i < 18; ++i) if (scanf("%x", bits + i) != 1) return 3;
        query_case(operation, kind, bits);
    }
    return ferror(stdin) ? 4 : 0;
}
