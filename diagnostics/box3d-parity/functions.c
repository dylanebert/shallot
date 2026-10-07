#include "box3d/math_functions.h"
#include "box3d/collision.h"
#include "simd.h"
#include "manifold.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(B3_SIMD_SSE2) && !defined(B3_SIMD_NEON)
#error This harness requires SSE2 (oracle authority) or NEON (native-only evidence).
#endif

static float value(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
static void emit(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    printf(" %08x", bits);
}
static void word(uint32_t u) { printf(" %08x", u); }
static b3Vec3 vec(const uint32_t* r, int i) { return (b3Vec3){value(r[i]), value(r[i+1]), value(r[i+2])}; }
static b3Quat quat(const uint32_t* r, int i) { return (b3Quat){value(r[i]), value(r[i+1]), value(r[i+2]), value(r[i+3])}; }
static void vector(b3Vec3 v) { emit(v.x); emit(v.y); emit(v.z); }
static b3Sweep sweep(const uint32_t* r, int i) {
    return (b3Sweep){vec(r,i),vec(r,i+3),vec(r,i+6),quat(r,i+9),quat(r,i+13)};
}
int main(void) {
    unsigned operation, count;
    uint32_t r[4096];
    while (scanf("%u %u", &operation, &count) == 2) {
        if (count > 4096) return 2;
        for (unsigned i = 0; i < count; ++i) {
            if (scanf("%x", &r[i]) != 1) return 3;
        }
        printf("%u", operation);
        switch (operation) {
            case 0: {
                float length;
                b3Vec3 n = b3GetLengthAndNormalize(&length,
                    (b3Vec3){value(r[0]), value(r[1]), value(r[2])});
                emit(n.x); emit(n.y); emit(n.z); emit(length);
                break;
            }
            case 1:
                emit(b3UnwindAngle(value(r[0])));
                break;
            case 2: {
                b3FloatW a = b3SetW(value(r[0]), value(r[1]), value(r[2]), value(r[3]));
                b3FloatW b = b3SetW(value(r[4]), value(r[5]), value(r[6]), value(r[7]));
                float out[4];
                b3StoreW(out, b3SymClampW(a, b));
                for (int i = 0; i < 4; ++i) emit(out[i]);
                break;
            }
            case 3: case 4: case 5: {
                b3Vec3 pa[128], pb[128];
                if (r[0] > 128 || r[2] > 128) return 6;
                for (unsigned i=0; i<r[0]; i++) pa[i]=vec(r,32+3*i);
                for (unsigned i=0; i<r[2]; i++) pb[i]=vec(r,416+3*i);
                b3ShapeProxy a={pa,(int)r[0],value(r[1])}, b={pb,(int)r[2],value(r[3])};
                b3Transform transform={vec(r,4),quat(r,7)};
                if (operation == 3) {
                    b3SimplexCache cache={0};
                    cache.metric=value(r[16]); cache.count=(uint16_t)r[17];
                    for(int i=0;i<4;i++) { cache.indexA[i]=(uint8_t)r[18+i]; cache.indexB[i]=(uint8_t)r[22+i]; }
                    b3DistanceInput input={a,b,transform,r[11]!=0};
                    b3DistanceOutput d=b3ShapeDistance(&input,&cache,NULL,0);
                    vector(d.pointA); vector(d.pointB); vector(d.normal); emit(d.distance); word(d.iterations);
                    emit(cache.metric); word(cache.count);
                    for(int i=0;i<4;i++) word(cache.indexA[i]);
                    for(int i=0;i<4;i++) word(cache.indexB[i]);
                } else if(operation == 4) {
                    b3ShapeCastPairInput input={a,b,transform,vec(r,12),value(r[15]),r[11]!=0};
                    b3CastOutput c=b3ShapeCast(&input);
                    word(c.hit); word(c.iterations);
                    if(c.hit) { emit(c.fraction); vector(c.point); vector(c.normal); word(c.triangleIndex); word(c.childIndex); word(c.materialIndex); }
                } else {
                    b3TOIInput input={a,b,sweep(r,800),sweep(r,817),value(r[15])};
                    b3TOIOutput t=b3TimeOfImpact(&input);
                    word(t.state); emit(t.fraction); emit(t.distance); vector(t.point); vector(t.normal);
                    word(t.distanceIterations); word(t.pushBackIterations); word(t.rootIterations); word(t.usedFallback);
                }
                break;
            }
            case 99: {
                b3BoxHull a=b3MakeBoxHull(1,1,1), b=b3MakeBoxHull(0.6f,1.2f,0.8f);
                word(sizeof(a)/4);
                for(size_t i=0;i<sizeof(a)/4;i++) { uint32_t u; memcpy(&u,(char*)&a+4*i,4); word(u); }
                word(sizeof(b)/4);
                for(size_t i=0;i<sizeof(b)/4;i++) { uint32_t u; memcpy(&u,(char*)&b+4*i,4); word(u); }
                break;
            }
            case 10: case 11: case 12: case 13: case 14: case 15: case 16: case 17: case 18: {
                b3BoxHull ha=b3MakeBoxHull(1,1,1), hb=b3MakeBoxHull(0.6f,1.2f,0.8f);
                b3Sphere sa={vec(r,12),value(r[15])}, sb={vec(r,16),value(r[19])};
                b3Capsule ca={vec(r,20),vec(r,23),value(r[26])}, cb={vec(r,27),vec(r,30),value(r[33])};
                b3Vec3 tri[3]={vec(r,34),vec(r,37),vec(r,40)};
                b3Transform xf={vec(r,4),quat(r,7)};
                b3SimplexCache cache={0};
                cache.metric=value(r[50]); cache.count=(uint16_t)r[51];
                for(int i=0;i<4;i++) { cache.indexA[i]=(uint8_t)r[52+i]; cache.indexB[i]=(uint8_t)r[56+i]; }
                b3SATCache sat={value(r[44]),(uint8_t)r[45],(uint8_t)r[46],(uint8_t)r[47],(uint8_t)r[48]};
                b3LocalManifoldPoint points[128]={0};
                b3LocalManifold m={0}; m.points=points; m.pointCount=(int)r[1]; m.feature=r[2]; m.squaredDistance=value(r[3]);
                int capacity=(int)r[0];
                switch(operation) {
                    case 10: b3CollideSpheres(&m,capacity,&sa,&sb,xf); break;
                    case 11: b3CollideCapsuleAndSphere(&m,capacity,&ca,&sb,xf); break;
                    case 12: b3CollideHullAndSphere(&m,capacity,&ha.base,&sb,xf,&cache); break;
                    case 13: b3CollideCapsules(&m,capacity,&ca,&cb,xf); break;
                    case 14: b3CollideHullAndCapsule(&m,capacity,&ha.base,&cb,xf,&cache); break;
                    case 15: b3CollideHulls(&m,capacity,&ha.base,&hb.base,xf,&sat); break;
                    case 16: b3CollideTriangleAndSphere(&m,capacity,tri,&sa); break;
                    case 17: b3CollideTriangleAndCapsule(&m,capacity,tri,&ca,&cache); break;
                    case 18: b3CollideTriangleAndHull(&m,capacity,tri[0],tri[1],tri[2],0,&ha.base,&sat,true); break;
                }
                word(m.pointCount); vector(m.normal); word(m.feature); emit(m.squaredDistance);
                for(int i=0;i<m.pointCount;i++) { vector(points[i].point); emit(points[i].separation); word(b3MakeFeatureId(points[i].pair)); word(points[i].triangleIndex); }
                emit(cache.metric); word(cache.count);
                for(int i=0;i<4;i++) word(cache.indexA[i]);
                for(int i=0;i<4;i++) word(cache.indexB[i]);
                emit(sat.separation); word(sat.type); word(sat.indexA); word(sat.indexB); word(sat.hit);
                break;
            }
            default: return 4;
        }
        putchar('\n');
    }
    return ferror(stdin) ? 5 : 0;
}
