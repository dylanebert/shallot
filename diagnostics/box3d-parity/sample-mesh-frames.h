// SPDX-License-Identifier: MIT
static std::vector<b3ShapeId> frameShapes;
static b3ShapeId FrameMeshShape(b3BodyId body,const b3ShapeDef* def,const b3MeshData* mesh,b3Vec3 scale) {
    b3ShapeId shape=RecordMeshShape(body,def,mesh,scale);frameShapes.push_back(shape);return shape;
}
static b3ShapeId FrameHeightFieldShape(b3BodyId body,const b3ShapeDef* def,const b3HeightFieldData* field) {
    b3ShapeId shape=b3CreateHeightFieldShape(body,def,field);frameShapes.push_back(shape);return shape;
}
void RecordGeometryFrame(b3WorldId) {
    const char* path=getenv("MESH_FRAMES");if(!path)return;
    FILE* file=fopen(path,"a");assert(file);
    static std::set<std::string> seen;
    struct Record { b3ShapeId shape; std::string id; b3Vec3 scale; };
    std::vector<Record> records;
    for(b3ShapeId shape:frameShapes) {
        if(!b3Shape_IsValid(shape))continue;
        const void* data;int size;const char* kind;b3Vec3 scale=b3Vec3_one;
        if(b3Shape_GetType(shape)==b3_meshShape) {
            b3Mesh mesh=b3Shape_GetMesh(shape);data=mesh.data;size=mesh.data->byteCount;scale=mesh.scale;kind="mesh";
        } else { const b3HeightFieldData* field=b3Shape_GetHeightField(shape);data=field;size=field->byteCount;kind="height"; }
        uint64_t hash;memcpy(&hash,(const char*)data+8,8);
        char id[40];snprintf(id,sizeof(id),"%s-%016llx",kind,(unsigned long long)hash);
        if(seen.insert(id).second) {
            fprintf(file,"{\"id\":\"%s\",\"kind\":\"%s\",\"bytes\":\"",id,kind);
            for(int i=0;i<size;++i)fprintf(file,"%02x",((const unsigned char*)data)[i]);
            fprintf(file,"\"}\n");
        }
        records.push_back({shape,id,scale});
    }
    fprintf(file,"{\"frame\":[");
    for(size_t i=0;i<records.size();++i) {
        const auto& r=records[i];b3BodyId body=b3Shape_GetBody(r.shape);
        b3Pos p=b3Body_GetPosition(body);b3Quat q=b3Body_GetRotation(body);
        fprintf(file,"%s{\"id\":\"%s\",\"p\":[%.17g,%.17g,%.17g],\"q\":[%.9g,%.9g,%.9g,%.9g],\"scale\":[%.9g,%.9g,%.9g]}",i?",":"",r.id.c_str(),p.x,p.y,p.z,q.v.x,q.v.y,q.v.z,q.s,r.scale.x,r.scale.y,r.scale.z);
    }
    fprintf(file,"]}\n");fclose(file);
}
#undef b3CreateMeshShape
#define b3CreateMeshShape FrameMeshShape
#define b3CreateHeightFieldShape FrameHeightFieldShape
