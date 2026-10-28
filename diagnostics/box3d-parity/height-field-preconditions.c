// SPDX-License-Identifier: MIT
#define B3_ENABLE_ASSERT 1
#include "height_field.c"
#include <stdio.h>
#include <stdlib.h>

static int Refuse(const char* condition, const char* file, int line) {
    (void)file;
    fprintf(stderr, "%d: %s\n", line, condition);
    exit(77);
}
int b3InternalAssert(const char* condition, const char* file, int line) {
    return Refuse(condition, file, line);
}
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    float heights[100] = {0};
    b3HeightFieldDef def = {0};
    def.countX = atoi(argv[1]);
    def.countZ = atoi(argv[2]);
    def.globalMinimumHeight = atof(argv[3]);
    def.globalMaximumHeight = atof(argv[4]);
    def.heights = heights;
    def.scale = (b3Vec3){2, 0, 2};
    b3HeightFieldData* field = b3CreateHeightField(&def);
    b3DestroyHeightField(field);
    return 0;
}
