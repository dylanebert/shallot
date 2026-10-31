#include "box3d/box3d.h"
#include <stdio.h>
static int calls;
static bool filter(b3ShapeId a,b3ShapeId b,void* ctx){(void)a;(void)b;(void)ctx;++calls;return true;}
int main(void){int n,mode;while(scanf("%d %d",&n,&mode)==2){b3WorldDef wd=b3DefaultWorldDef();wd.gravity=(b3Vec3){0};wd.enableSleep=false;b3WorldId world=b3CreateWorld(&wd);for(int i=0;i<n;i++){b3BodyDef bd=b3DefaultBodyDef();bd.type=b3_dynamicBody;b3BodyId body=b3CreateBody(world,&bd);b3ShapeDef sd=b3DefaultShapeDef();sd.enableCustomFiltering=mode==2||(mode==1&&i==0);b3Sphere sphere={{0},0.5f};b3CreateSphereShape(body,&sd,&sphere);}calls=0;if(mode)b3World_SetCustomFilterCallback(world,filter,NULL);b3World_Step(world,0.0f,1);b3Counters c=b3World_GetCounters(world);printf("{\"count\":%d,\"mode\":%d,\"contacts\":%d,\"calls\":%d}\n",n,mode,c.contactCount,calls);b3DestroyWorld(world);}return 0;}
