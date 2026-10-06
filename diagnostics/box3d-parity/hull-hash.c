#include "box3d/box3d.h"
#include "core.h"
#include "physics_world.h"
#include <stdio.h>
#include <stdlib.h>

static int allocations;
static void* count_alloc(int32_t size, int32_t alignment)
{
    ++allocations;
    return aligned_alloc(alignment, (size + alignment - 1) & ~(alignment - 1));
}
static void count_free(void* memory) { free(memory); }

static void emit(const b3HullData* hull)
{
    fwrite(hull, 1, hull->byteCount, stdout);
}

int main(void)
{
    b3BoxHull unit = b3MakeBoxHull(1.0f, 1.0f, 1.0f);
    b3BoxHull oblong = b3MakeBoxHull(0.5f, 1.0f, 2.0f);
    b3BoxHull offset = b3MakeOffsetBoxHull(0.75f, 1.25f, 2.5f, (b3Vec3){1.0f, -2.0f, 3.0f});
    emit(&unit.base);
    emit(&oblong.base);
    emit(&offset.base);
    b3Vec3 points[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    b3HullData* tetra = b3CreateHull(points, 4, 4);
    if (tetra == NULL) return 1;
    emit(tetra);
    b3DestroyHull(tetra);
    b3SetAllocator(count_alloc, count_free);
    b3WorldDef def = b3DefaultWorldDef();
    b3WorldId id = b3CreateWorld(&def);
    b3World* world = b3GetWorldFromId(id);
    const b3HullData* warm = b3AddHullToDatabase(world, &unit.base);
    b3RemoveHullFromDatabase(world, warm);
    int before = allocations;
    const b3HullData* first = b3AddHullToDatabase(world, &unit.base);
    int miss = allocations - before;
    before = allocations;
    const b3HullData* second = b3AddHullToDatabase(world, &unit.base);
    int hit = allocations - before;
    before = allocations;
    b3RemoveHullFromDatabase(world, first);
    b3RemoveHullFromDatabase(world, second);
    int destroy = allocations - before;
    fprintf(stderr, "miss=%d hit=%d destroy=%d\n", miss, hit, destroy);
    b3DestroyWorld(id);
    return ferror(stdout) ? 1 : 0;
}
