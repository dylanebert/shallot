// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <cstdio>
#include <cstring>
// MESH_LOADER

int main(int argc, char** argv)
{
    if (argc != 3) return 1;
    bool median = strstr(argv[1], "conveyor.obj") != nullptr;
    if (strcmp(argv[2], "build") == 0)
    {
        b3MeshData* mesh = CreateMeshData(argv[1], 1.0f, false, median, true, true);
        if (mesh == nullptr) return 2;
        fwrite(mesh, 1, mesh->byteCount, stdout);
        DestroyMeshData(mesh);
        return 0;
    }
    TempMesh mesh;
    if (!LoadTempMesh(argv[1], &mesh, 1.0f, false)) return 2;
    if (median)
    {
        b3MeshData* built = CreateMeshData(argv[1], 1.0f, false, true, true, true);
        if (built == nullptr) return 2;
        const b3MeshTriangle* triangles = b3GetMeshTriangles(built);
        const int pairs[][2] = {{0,4},{9,12},{21,38},{43,46},{30,33},{18,24}};
        memset(mesh.materialIndices.data(), 0, mesh.materialIndices.size());
        for (int role = 0; role < 6; ++role)
            for (int index : pairs[role])
                for (int i = 0; i < int(mesh.materialIndices.size()); ++i)
                    if (mesh.indices[3*i] == triangles[index].index1 &&
                        mesh.indices[3*i+1] == triangles[index].index2 &&
                        mesh.indices[3*i+2] == triangles[index].index3)
                        mesh.materialIndices[i] = uint8_t(role + 1);
        DestroyMeshData(built);
    }
    int counts[] = {int(mesh.vertices.size()), int(mesh.indices.size()), int(mesh.materialIndices.size())};
    fwrite(counts, sizeof(int), 3, stdout);
    fwrite(mesh.vertices.data(), sizeof(b3Vec3), mesh.vertices.size(), stdout);
    fwrite(mesh.indices.data(), sizeof(int), mesh.indices.size(), stdout);
    fwrite(mesh.materialIndices.data(), sizeof(uint8_t), mesh.materialIndices.size(), stdout);
}
