// SPDX-License-Identifier: MIT
// Native Box3D authoring images, including relative offsets, padding and construction hashes.
#include "box3d/collision.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static int first=1;
static void floats(const float* a,int n){printf("[");for(int i=0;i<n;++i){uint32_t u;memcpy(&u,a+i,4);printf("%s%u",i?",":"",u);}printf("]");}
static void ints(const int* a,int n){printf("[");for(int i=0;i<n;++i)printf("%s%d",i?",":"",a[i]);printf("]");}
static void bytes(const uint8_t* a,int n){printf("[");if(a)for(int i=0;i<n;++i)printf("%s%u",i?",":"",(unsigned)a[i]);printf("]");}
static void head(const char* name,int op,const float* args,int n){printf("%s{\"name\":\"%s\",\"operation\":%d,\"args\":",first?"":",\n",name,op);first=0;floats(args,n);}
static void image(const void* p,int size){printf(",\"words\":");if(!p){printf("null}");return;}printf("[");for(int i=0;i<size/4;++i){uint32_t u;memcpy(&u,(const char*)p+4*i,4);printf("%s%u",i?",":"",u);}printf("]}");}
static void mesh(const char* name,int op,float* args,int n,b3MeshData* m){head(name,op,args,n);image(m,m?m->byteCount:0);if(m)b3DestroyMesh(m);}
static void field(const char* name,int op,float* args,int n,b3HeightFieldData* h){head(name,op,args,n);image(h,h?h->byteCount:0);if(h)b3DestroyHeightField(h);}
static void custom(const char* name,b3Vec3* v,int nv,int* ix,int ni,uint8_t* mat,bool weld,float tol,bool median,bool edges){
 b3MeshDef d={0};d.vertices=v;d.vertexCount=nv;d.indices=ix;d.triangleCount=ni/3;d.materialIndices=mat;d.weldVertices=weld;d.weldTolerance=tol;d.useMedianSplit=median;d.identifyEdges=edges;
 float a[]={tol,(float)weld,(float)median,(float)edges};head(name,0,a,4);printf(",\"vertices\":");floats((const float*)v,3*nv);printf(",\"indices\":");ints(ix,ni);printf(",\"materials\":");bytes(mat,ni/3);b3MeshData* m=b3CreateMesh(&d,NULL,0);image(m,m?m->byteCount:0);if(m)b3DestroyMesh(m);
}
static void custom_field(const char* name,int nx,int nz,b3Vec3 scale,float lo,float hi,bool cw,float* heights,uint8_t* mat){
 b3HeightFieldDef d={0};d.countX=nx;d.countZ=nz;d.scale=scale;d.globalMinimumHeight=lo;d.globalMaximumHeight=hi;d.clockwiseWinding=cw;d.heights=heights;d.materialIndices=mat;
 float a[]={(float)nx,(float)nz,scale.x,scale.y,scale.z,lo,hi,(float)cw};head(name,10,a,8);printf(",\"heights\":");floats(heights,nx*nz);printf(",\"materials\":");bytes(mat,(nx-1)*(nz-1));b3HeightFieldData* h=b3CreateHeightField(&d);image(h,h->byteCount);b3DestroyHeightField(h);
}
int main(void){puts("[");
 b3Vec3 v[]={{0,0,0},{1,0,0},{0,0,1},{1,0,1},{0,0,0},{1,0,0},{0,0,1},{0.000001f,0,0},{2,0,0}};
 int ix[]={0,2,1,1,2,3,4,6,5,7,2,1,0,1,8,0,0,1};uint8_t mat[]={1,2,3,4,5,6};
 custom("raw",v,9,ix,18,mat,false,0,false,true);custom("weld",v,9,ix,18,mat,true,0.001f,false,true);custom("median",v,9,ix,18,NULL,false,0,true,false);
 int same[]={0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1};custom("sah-unsplit-leaf",v,3,same,18,NULL,false,0,false,true);
 int half[36];for(int i=0;i<36;++i)half[i]=same[i%18];custom("sah-half",v,3,half,36,NULL,false,0,false,true);
 custom("empty",v,0,ix,0,NULL,false,0,false,false);int deg[]={0,0,1,0,1,8};custom("all-degenerate",v,9,deg,6,NULL,false,0,false,false);
 b3Vec3 crease[]={{0,0,0},{1,0,0},{0,0,1},{1,1,1},{1,-1,1}};int ci[]={0,2,1,1,2,3,1,2,4};custom("non-manifold",crease,5,ci,9,NULL,false,0,false,true);custom("crease",crease,4,ci,6,NULL,false,0,false,true);
 float box[]={0.5f,-0.25f,1,2,1,0.5f,1};mesh("box",1,box,7,b3CreateBoxMesh((b3Vec3){box[0],box[1],box[2]},(b3Vec3){box[3],box[4],box[5]},true));box[6]=0;mesh("box-no-edges",1,box,7,b3CreateBoxMesh((b3Vec3){0.5f,-0.25f,1},(b3Vec3){2,1,0.5f},false));mesh("hollow",2,box,6,b3CreateHollowBoxMesh((b3Vec3){0.5f,-0.25f,1},(b3Vec3){2,1,0.5f}));
 float grid[]={4,5,0.3f,3,1};mesh("grid",3,grid,5,b3CreateGridMesh(4,5,0.3f,3,true));float grid2[]={1,2,1,0,0};mesh("grid-small",3,grid2,5,b3CreateGridMesh(1,2,1,0,false));
 float wave[]={6,5,1,0.4f,0.05f,0.1f};mesh("wave",4,wave,6,b3CreateWaveMesh(6,5,1,0.4f,0.05f,0.1f));float wave2[]={9,7,0.3f,-1.5f,0.2f,0.13f};mesh("wave-negative",4,wave2,6,b3CreateWaveMesh(9,7,0.3f,-1.5f,0.2f,0.13f));
 float torus[]={8,6,3,1};mesh("torus",5,torus,4,b3CreateTorusMesh(8,6,3,1));float torus2[]={3,4,0.5f,0.1f};mesh("torus-small",5,torus2,4,b3CreateTorusMesh(3,4,0.5f,0.1f));
 float h[20]={-2,0.125f,1,3,-0.8f, 0.3f,2,-1,0,1, 1,-2,0.6f,2,0, 0.2f,0.4f,-0.2f,1.3f,2};uint8_t hm[]={0,1,2,3,4,255,6,7,8,9,10,11};
 custom_field("height-clamped",5,4,(b3Vec3){0.3f,1.5f,2},-1,2,false,h,hm);custom_field("height-clockwise",5,4,(b3Vec3){0.3f,1.5f,2},-1,2,true,h,hm);custom_field("height-no-materials",5,4,(b3Vec3){1,1,1},-256,256,false,h,NULL);custom_field("height-zero-range",5,4,(b3Vec3){1,1,1},1,1,false,h,NULL);
 float hg[]={6,7,0.3f,1.5f,2,1};field("height-grid-holes",11,hg,6,b3CreateGrid(6,7,(b3Vec3){0.3f,1.5f,2},true));float hg2[]={2,2,1,1,1,0};field("height-grid-minimum",11,hg2,6,b3CreateGrid(2,2,(b3Vec3){1,1,1},false));
 float hw[]={32,32,2,1.5f,2,0.1f,0.03333f,0};field("height-wave",12,hw,8,b3CreateWave(32,32,(b3Vec3){2,1.5f,2},0.1f,0.03333f,false));float hw2[]={6,8,1,0.5f,1,0.25f,0.5f,1};field("height-wave-holes",12,hw2,8,b3CreateWave(6,8,(b3Vec3){1,0.5f,1},0.25f,0.5f,true));
 puts("\n]");return 0;
}
