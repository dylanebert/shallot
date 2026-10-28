// SPDX-License-Identifier: MIT
static b3ShapeId RecordMeshShape(b3BodyId body, const b3ShapeDef* def, const b3MeshData* mesh, b3Vec3 scale)
{
    const char* path = getenv("MESH_GEOMETRY");
    if (path) {
        FILE* file = fopen(path, "a"); assert(file);
        b3Pos p = b3Body_GetPosition(body);
        b3Quat q = b3Body_GetRotation(body);
        fprintf(file, "{\"p\":[%.17g,%.17g,%.17g],\"q\":[%.9g,%.9g,%.9g,%.9g],\"scale\":[%.9g,%.9g,%.9g],\"bytes\":\"", p.x,p.y,p.z,q.v.x,q.v.y,q.v.z,q.s,scale.x,scale.y,scale.z);
        const unsigned char* bytes = (const unsigned char*)mesh;
        for (int i=0;i<mesh->byteCount;++i) fprintf(file,"%02x",bytes[i]);
        fprintf(file,"\"}\n"); fclose(file);
    }
    return b3CreateMeshShape(body,def,mesh,scale);
}
#undef b3CreateMeshShape
#define b3CreateMeshShape RecordMeshShape
