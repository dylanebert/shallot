// SPDX-License-Identifier: MIT
#include "box3d/collision.h"
#include "shape.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
static uint32_t input[64], output[16];
static int first = 1;
static void f(int i, float x) { memcpy(input+i, &x, 4); }
static void v(int i, b3Vec3 x) { f(i,x.x); f(i+1,x.y); f(i+2,x.z); }
static void emit(int op, int n) {
 printf("%s{\"operation\":%d,\"input\":[", first?"":",\n",op); first=0;
 for(int i=0;i<n;++i)printf("%s%u",i?",":"",input[i]);
 printf("],\"output\":[");for(int i=0;i<16;++i)printf("%s%u",i?",":"",output[i]);printf("]}");
}
static void box(int op, b3AABB a) { memset(output,0,sizeof output); memcpy(output,&a,24); emit(op,32); }
static void tri(int op,b3Triangle t,int n) { memset(output,0,sizeof output); memcpy(output,&t,sizeof t); emit(op,n); }
int main(void) {
 puts("[");
 for(int sample=0;sample<4;++sample) {
  memset(input,0,sizeof input);
  b3Transform pose = { {0.3f,-2.1f,4.7f}, {{0,0,0},1} };
  if(sample&1) pose.q = b3MakeQuatFromAxisAngle(b3Normalize((b3Vec3){1,2,-3}),1.3f);
  if(sample&2) pose.p = (b3Vec3){0,0,0};
  v(0,pose.p);v(3,pose.q.v);f(6,pose.q.s);
  b3Sphere s={{-0.7f,1.2f,2.3f},0.3f};v(8,s.center);f(11,s.radius);box(0,b3ComputeSphereAABB(&s,pose));
  b3Capsule c={{-0.7f,1.2f,2.3f},{0.2f,-0.4f,1.1f},0.3f};if(sample&2)c.center2=c.center1;
  v(8,c.center1);v(11,c.center2);f(14,c.radius);box(1,b3ComputeCapsuleAABB(&c,pose));
  b3HullData h={0};h.aabb=(b3AABB){{-0.7f,-1.2f,-2.3f},{1.1f,2.4f,3.8f}};
  v(8,h.aabb.lowerBound);v(11,h.aabb.upperBound);box(2,b3ComputeHullAABB(&h,pose));
  b3MeshData m={0};m.bounds=h.aabb;
  b3Vec3 scale={(sample&1)?-0.3f:0.3f,1.7f,(sample&2)?-2.1f:2.1f};v(14,scale);box(3,b3ComputeMeshAABB(&m,pose,scale));
 }
 b3Vec3 scales[]={{0,-0.0f,1},{-0.00001f,0.009f,-3},{0.01f,-0.01f,0.3f}};
 for(int i=0;i<3;++i) {v(8,scales[i]);memset(output,0,sizeof output);b3Vec3 s=b3SafeScale(scales[i]);memcpy(output,&s,12);emit(4,32);}
 struct { b3MeshData data; b3Vec3 vertices[3]; int indices[3]; uint8_t flags[1]; } mesh={0};
 mesh.data.vertexOffset=(char*)mesh.vertices-(char*)&mesh;mesh.data.triangleOffset=(char*)mesh.indices-(char*)&mesh;mesh.data.flagsOffset=(char*)mesh.flags-(char*)&mesh;
 mesh.vertices[0]=(b3Vec3){0.3f,-1.1f,2.7f};mesh.vertices[1]=(b3Vec3){-2.2f,3.3f,0.4f};mesh.vertices[2]=(b3Vec3){1.7f,0.1f,-0.9f};mesh.indices[0]=0;mesh.indices[1]=1;mesh.indices[2]=2;
 int flags[]={0,1,2,4,16,32,64,119};
 for(int flip=0;flip<2;++flip)for(int fi=0;fi<8;++fi) {
  int flag=flags[fi];b3Mesh m={&mesh.data,{flip?-0.3f:0.3f,1.7f,2.1f}};mesh.flags[0]=flag;
  for(int i=0;i<3;++i)v(8+3*i,mesh.vertices[i]);input[17]=0;input[18]=1;input[19]=2;input[20]=flag;v(21,m.scale);tri(5,b3GetMeshTriangle(&m,0),32);
 }
 struct { b3HeightFieldData data; uint16_t heights[12]; uint8_t flags[12]; } field={0};
 field.data.columnCount=4;field.data.rowCount=3;field.data.minHeight=-1.3f;field.data.heightScale=0.00013f;field.data.scale=(b3Vec3){0.3f,1.7f,2.1f};
 field.data.heightsOffset=(char*)field.heights-(char*)&field;field.data.flagsOffset=(char*)field.flags-(char*)&field;
 for(int i=0;i<12;++i)field.heights[i]=(i*5687)%65536;
 for(int cw=0;cw<2;++cw)for(int i=0;i<12;++i)for(int fi=0;fi<8;++fi) {
  int flag=flags[fi];field.data.clockwise=cw;field.flags[i]=flag;input[8]=4;input[9]=i;f(10,field.data.minHeight);f(11,field.data.heightScale);v(12,field.data.scale);input[15]=cw;input[16]=flag;
  for(int j=0;j<12;++j)input[32+j]=field.heights[j];tri(6,b3GetHeightFieldTriangle(&field.data,i),44);
 }
 puts("\n]");
}
