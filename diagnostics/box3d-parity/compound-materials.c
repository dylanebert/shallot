#include "box3d/box3d.h"
#include <stdio.h>

int main(void)
{
    b3Vec3 vertices[18];
    int indices[18];
    uint8_t materialIndices[6];
    b3SurfaceMaterial materials[6];
    for (int i = 0; i < 6; ++i)
    {
        vertices[3*i] = (b3Vec3){3.0f*i, 0, 0};
        vertices[3*i+1] = (b3Vec3){3.0f*i, 0, 1};
        vertices[3*i+2] = (b3Vec3){3.0f*i+1, 0, 0};
        for (int j = 0; j < 3; ++j) indices[3*i+j] = 3*i+j;
        materialIndices[i] = (uint8_t)i;
        materials[i] = b3DefaultSurfaceMaterial();
        materials[i].userMaterialId = 100+i;
    }
    b3MeshDef md = {0};
    md.vertices = vertices; md.vertexCount = 18; md.indices = indices;
    md.triangleCount = 6; md.materialIndices = materialIndices;
    b3MeshData* mesh = b3CreateMesh(&md, NULL, 0);
    b3CompoundMeshDef child = {0};
    child.meshData = mesh; child.transform = b3Transform_identity;
    child.scale = (b3Vec3){1,1,1}; child.materials = materials; child.materialCount = 6;
    b3CompoundDef def = {0}; def.meshes = &child; def.meshCount = 1;
    b3CompoundData* compound = b3CreateCompound(&def);
    printf("%d", compound->materialCount);
    b3WorldDef wd = b3DefaultWorldDef(); b3WorldId world = b3CreateWorld(&wd);
    b3BodyDef bd = b3DefaultBodyDef(); b3BodyId body = b3CreateBody(world, &bd);
    b3ShapeDef sd = b3DefaultShapeDef(); b3CreateBakedCompoundShape(body, &sd, compound);
    for (int i = 0; i < 6; ++i)
    {
        b3RayResult hit = b3World_CastRayClosest(world, (b3Pos){3.0*i+0.2, 1, 0.2}, (b3Vec3){0,-2,0}, b3DefaultQueryFilter());
        if (!hit.hit) return 2;
        printf(" %llu", (unsigned long long)hit.userMaterialId);
    }
    printf("\n");
    b3DestroyWorld(world);
    b3DestroyCompound(compound); b3DestroyMesh(mesh);
    return 0;
}
