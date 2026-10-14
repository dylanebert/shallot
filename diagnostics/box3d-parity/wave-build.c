// SPDX-License-Identifier: MIT
#include "box3d/collision.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    b3MeshData* mesh = b3CreateWaveMesh(6, 5, 1.0f, 0.4f, 0.05f, 0.1f);
    const b3Vec3* vertices = b3GetMeshVertices(mesh);
    printf("{\"meshHeights\":[");
    for (int i = 0; i < mesh->vertexCount; ++i) {
        uint32_t bits;
        memcpy(&bits, &vertices[i].y, sizeof(bits));
        printf("%s%u", i ? "," : "", bits);
    }
    b3DestroyMesh(mesh);
    b3HeightFieldData* field = b3CreateWave(32, 32, (b3Vec3){2.0f, 1.5f, 2.0f}, 0.1f, 0.03333f, false);
    const uint16_t* heights = b3GetHeightFieldCompressedHeights(field);
    printf("],\"fieldHeights\":[");
    for (int i = 0; i < field->columnCount * field->rowCount; ++i) {
        printf("%s%u", i ? "," : "", (unsigned)heights[i]);
    }
    printf("]}\n");
    b3DestroyHeightField(field);
    return 0;
}
