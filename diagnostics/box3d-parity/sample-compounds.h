// SPDX-License-Identifier: MIT
static void CompoundVector(FILE* file, b3Vec3 v)
{
    fprintf(file, "[%.9g,%.9g,%.9g]", v.x, v.y, v.z);
}
static void CompoundTransform(FILE* file, b3Transform t)
{
    fprintf(file, "{\"p\":"); CompoundVector(file, t.p);
    fprintf(file, ",\"q\":[%.9g,%.9g,%.9g,%.9g]}", t.q.v.x, t.q.v.y, t.q.v.z, t.q.s);
}
static b3CompoundData* RecordCompound(const b3CompoundDef* def)
{
    b3CompoundData* data = b3CreateCompound(def);
    const char* path = getenv("COMPOUND_GEOMETRY");
    if (!path) return data;
    FILE* file = fopen(path, "a"); assert(file);
    fprintf(file, "[");
    bool first = true;
    auto separator = [&]() { if (!first) fprintf(file, ","); first = false; };
    for (int i = 0; i < data->capsuleCount; ++i) {
        separator(); auto child = b3GetCompoundCapsule(data, i);
        fprintf(file, "{\"kind\":\"capsule\",\"center1\":"); CompoundVector(file, child.capsule.center1);
        fprintf(file, ",\"center2\":"); CompoundVector(file, child.capsule.center2);
        fprintf(file, ",\"radius\":%.9g}", child.capsule.radius);
    }
    for (int i = 0; i < data->hullCount; ++i) {
        separator(); auto child = b3GetCompoundHull(data, i);
        fprintf(file, "{\"kind\":\"hull\",\"transform\":"); CompoundTransform(file, child.transform);
        fprintf(file, ",\"points\":[");
        const b3Vec3* points = b3GetHullPoints(child.hull);
        for (int j = 0; j < child.hull->vertexCount; ++j) { if (j) fprintf(file, ","); CompoundVector(file, points[j]); }
        fprintf(file, "]}");
    }
    for (int i = 0; i < data->meshCount; ++i) {
        separator(); auto child = b3GetCompoundMesh(data, i);
        fprintf(file, "{\"kind\":\"mesh\",\"transform\":"); CompoundTransform(file, child.transform);
        fprintf(file, ",\"scale\":"); CompoundVector(file, child.scale);
        fprintf(file, ",\"bytes\":\"");
        const unsigned char* bytes = (const unsigned char*)child.meshData;
        for (int j = 0; j < child.meshData->byteCount; ++j) fprintf(file, "%02x", bytes[j]);
        fprintf(file, "\"}");
    }
    for (int i = 0; i < data->sphereCount; ++i) {
        separator(); auto child = b3GetCompoundSphere(data, i);
        fprintf(file, "{\"kind\":\"sphere\",\"center\":"); CompoundVector(file, child.sphere.center);
        fprintf(file, ",\"radius\":%.9g}", child.sphere.radius);
    }
    fprintf(file, "]\n"); fclose(file);
    return data;
}
#define b3CreateCompound RecordCompound
