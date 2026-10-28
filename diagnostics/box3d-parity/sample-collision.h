// SPDX-License-Identifier: MIT
namespace Collision {
FILE* queries = nullptr;
void Vector(b3Vec3 v);
b3DistanceOutput Distance(const b3DistanceInput* input, b3SimplexCache* cache, b3Simplex* simplexes, int capacity);
b3CastOutput Cast(const b3ShapeCastPairInput* input);
b3TOIOutput TOI(const b3TOIInput* input);
void Float(float v) { unsigned bits; memcpy(&bits, &v, 4); fprintf(queries, " %08x", bits); }
void Vector(b3Vec3 v) { Float(v.x); Float(v.y); Float(v.z); }
b3DistanceOutput Distance(const b3DistanceInput* input, b3SimplexCache* cache, b3Simplex* simplexes, int capacity) {
    auto d = b3ShapeDistance(input,cache,simplexes,capacity);
    if (queries) {
        fprintf(queries,"distance"); Vector(d.pointA); Vector(d.pointB); Vector(d.normal); Float(d.distance);
        fprintf(queries," %d",d.iterations); Float(cache->metric); fprintf(queries," %d",cache->count);
        for(int i=0;i<4;i++) fprintf(queries," %d",cache->indexA[i]);
        for(int i=0;i<4;i++) fprintf(queries," %d",cache->indexB[i]);
        fprintf(queries," %d",d.simplexCount);
        for(int i=0;i<d.simplexCount;i++) { auto& s=simplexes[i]; fprintf(queries," %d",s.count); for(int j=0;j<s.count;j++) { auto& v=s.vertices[j]; Vector(v.wA); Vector(v.wB); Vector(v.w); Float(v.a); fprintf(queries," %d %d",v.indexA,v.indexB); } }
        fprintf(queries,"\n");
    }
    return d;
}
b3CastOutput Cast(const b3ShapeCastPairInput* input) {
    auto d=b3ShapeCast(input);
    if(queries) { fprintf(queries,"cast"); Vector(d.point); Vector(d.normal); Float(d.fraction); fprintf(queries," %d %d\n",d.iterations,d.hit); }
    return d;
}
b3TOIOutput TOI(const b3TOIInput* input) {
    auto d=b3TimeOfImpact(input);
    if(queries) { fprintf(queries,"toi %d",d.state); Float(d.fraction); Float(d.distance); Vector(d.point); Vector(d.normal); fprintf(queries," %d %d %d\n",d.distanceIterations,d.pushBackIterations,d.rootIterations); }
    return d;
}
void Hit(b3Pos p, b3Vec3 n, float fraction, uint64_t material, int triangle, int child) {
    fprintf(queries, "hit");
    for (float v : {float(p.x),float(p.y),float(p.z),n.x,n.y,n.z,fraction}) Float(v);
    fprintf(queries, " %llu %d %d", (unsigned long long)material, triangle, child);
}
struct Context { b3CastResultFcn* callback; void* user; };
float Callback(b3ShapeId id, b3Pos p, b3Vec3 n, float fraction, uint64_t material, int triangle, int child, void* context) {
    auto& c = *static_cast<Context*>(context);
    float result = c.callback(id,p,n,fraction,material,triangle,child,c.user);
    if (queries) { Hit(p,n,fraction,material,triangle,child); Float(result); fprintf(queries,"\n"); }
    return result;
}
void Ray(b3WorldId w, b3Pos p, b3Vec3 t, b3QueryFilter f, b3CastResultFcn* callback, void* context) {
    if (queries) fprintf(queries,"ray\n");
    Context c = {callback,context}; b3World_CastRay(w,p,t,f,Callback,&c);
    if (queries) fprintf(queries,"end\n");
}
void Shape(b3WorldId w, b3Pos p, const b3ShapeProxy* proxy, b3Vec3 t, b3QueryFilter f, b3CastResultFcn* callback, void* context) {
    if (queries) fprintf(queries,"shape\n");
    Context c = {callback,context}; b3World_CastShape(w,p,proxy,t,f,Callback,&c);
    if (queries) fprintf(queries,"end\n");
}
b3RayResult Closest(b3WorldId w, b3Pos p, b3Vec3 t, b3QueryFilter f) {
    auto r = b3World_CastRayClosest(w,p,t,f);
    if (queries) { fprintf(queries,"closest %d\n",r.hit); if(r.hit) { Hit(r.point,r.normal,r.fraction,r.userMaterialId,r.triangleIndex,r.childIndex); fprintf(queries,"\n"); } fprintf(queries,"end\n"); }
    return r;
}
struct OverlapContext { b3OverlapResultFcn* callback; void* user; };
bool OverlapCallback(b3ShapeId id, void* context) {
    auto& c = *static_cast<OverlapContext*>(context); bool result = c.callback(id,c.user);
    if (queries) fprintf(queries,"overlapHit %d\n",result);
    return result;
}
void Overlap(b3WorldId w, b3Pos p, const b3ShapeProxy* proxy, b3QueryFilter f, b3OverlapResultFcn* callback, void* context) {
    if (queries) fprintf(queries,"overlap\n");
    OverlapContext c = {callback,context}; b3World_OverlapShape(w,p,proxy,f,OverlapCallback,&c);
    if (queries) fprintf(queries,"end\n");
}
b3BodyCastResult BodyRay(b3BodyId body,b3Pos p,b3Vec3 t,b3QueryFilter f,float maxFraction,b3WorldTransform transform) {
    auto r = b3Body_CastRay(body,p,t,f,maxFraction,transform);
    if (queries) { fprintf(queries,"body %d\n",r.hit); if(r.hit) { Hit(r.point,r.normal,r.fraction,r.userMaterialId,r.triangleIndex,0); fprintf(queries,"\n"); } fprintf(queries,"end\n"); }
    return r;
}
}
