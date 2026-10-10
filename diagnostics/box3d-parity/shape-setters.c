// SPDX-License-Identifier: MIT
// Getter evidence for the live b3Shape_Set* boundaries exercised by shape-setters.oracle.ts.
#include "box3d/box3d.h"
#include "box3d/collision.h"

#include <stdio.h>

static b3WorldId makeWorld( void )
{
	b3WorldDef def = b3DefaultWorldDef();
	def.gravity = ( b3Vec3 ){ 0.0f, 0.0f, 0.0f };
	def.workerCount = 1;
	return b3CreateWorld( &def );
}

static b3BodyId makeBody( b3WorldId world, b3BodyType type )
{
	b3BodyDef def = b3DefaultBodyDef();
	def.type = type;
	return b3CreateBody( world, &def );
}

static b3MeshData* makeMesh( float x )
{
	b3Vec3 vertices[3] = { { x, 0.0f, 0.0f }, { x + 1.0f, 0.0f, 0.0f }, { x, 0.0f, 1.0f } };
	int32_t indices[3] = { 0, 2, 1 };
	b3MeshDef def = { 0 };
	def.vertices = vertices;
	def.indices = indices;
	def.vertexCount = 3;
	def.triangleCount = 1;
	def.weldTolerance = 0.0f;
	return b3CreateMesh( &def, NULL, 0 );
}

int main( void )
{
	b3WorldId world = makeWorld();
	b3ShapeDef def = b3DefaultShapeDef();
	b3BodyId dynamic = makeBody( world, b3_dynamicBody );
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.5f };
	b3ShapeId shape = b3CreateSphereShape( dynamic, &def, &sphere );

	b3Shape_SetDensity( shape, 2.75f, false );
	printf( "density %.9g\n", b3Shape_GetDensity( shape ) );
	b3Shape_SetFriction( shape, 0.37f );
	printf( "friction %.9g\n", b3Shape_GetFriction( shape ) );
	b3Shape_SetRestitution( shape, 0.61f );
	printf( "restitution %.9g\n", b3Shape_GetRestitution( shape ) );

	b3SurfaceMaterial material = b3DefaultSurfaceMaterial();
	material.friction = 0.23f;
	material.restitution = 0.34f;
	material.rollingResistance = 0.45f;
	material.tangentVelocity = ( b3Vec3 ){ 1.25f, -2.5f, 3.75f };
	material.userMaterialId = UINT64_C( 0x123456789abcdef0 );
	material.customColor = UINT32_C( 0x9a654321 );
	b3Shape_SetSurfaceMaterial( shape, material );
	material = b3Shape_GetSurfaceMaterial( shape );
	printf( "surface %.9g %.9g %.9g %.9g %.9g %.9g %llu %u\n", material.friction,
		material.restitution, material.rollingResistance, material.tangentVelocity.x,
		material.tangentVelocity.y, material.tangentVelocity.z,
		(unsigned long long)material.userMaterialId, material.customColor );

	b3Filter filter = b3DefaultFilter();
	filter.categoryBits = UINT64_C( 0x123456789abcdef0 );
	filter.maskBits = UINT64_C( 0xfedcba9876543210 );
	filter.groupIndex = -7;
	b3Shape_SetFilter( shape, filter, true );
	filter = b3Shape_GetFilter( shape );
	printf( "filter %llu %llu %d\n", (unsigned long long)filter.categoryBits,
		(unsigned long long)filter.maskBits, filter.groupIndex );

	b3Shape_EnableSensorEvents( shape, true );
	printf( "sensorEvents %d\n", (int)b3Shape_AreSensorEventsEnabled( shape ) );
	b3Shape_EnableContactEvents( shape, true );
	printf( "contactEvents %d\n", (int)b3Shape_AreContactEventsEnabled( shape ) );
	b3Shape_EnableHitEvents( shape, true );
	printf( "hitEvents %d\n", (int)b3Shape_AreHitEventsEnabled( shape ) );
	b3Shape_EnablePreSolveEvents( shape, true );
	printf( "preSolveEvents %d\n", (int)b3Shape_ArePreSolveEventsEnabled( shape ) );

	b3Sphere nextSphere = { { 0.25f, -0.5f, 0.75f }, 0.8f };
	b3Shape_SetSphere( shape, &nextSphere );
	nextSphere = b3Shape_GetSphere( shape );
	printf( "sphere %.9g %.9g %.9g %.9g\n", nextSphere.center.x, nextSphere.center.y,
		nextSphere.center.z, nextSphere.radius );

	b3BodyId capsuleBody = makeBody( world, b3_dynamicBody );
	b3Capsule capsule = { { 0.0f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f }, 0.25f };
	b3ShapeId capsuleShape = b3CreateCapsuleShape( capsuleBody, &def, &capsule );
	b3Capsule nextCapsule = { { -0.75f, 0.25f, 1.5f }, { 0.5f, 1.25f, -0.25f }, 0.4f };
	b3Shape_SetCapsule( capsuleShape, &nextCapsule );
	nextCapsule = b3Shape_GetCapsule( capsuleShape );
	printf( "capsule %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", nextCapsule.center1.x,
		nextCapsule.center1.y, nextCapsule.center1.z, nextCapsule.center2.x,
		nextCapsule.center2.y, nextCapsule.center2.z, nextCapsule.radius );

	b3Vec3 initialPoints[8] = { { -1, -1, -1 }, { 1, -1, -1 }, { 1, 1, -1 }, { -1, 1, -1 },
		{ -1, -1, 1 }, { 1, -1, 1 }, { 1, 1, 1 }, { -1, 1, 1 } };
	b3HullData* initialHull = b3CreateHull( initialPoints, 8, 8 );
	b3HullData* nextHull = b3CreateHull( (b3Vec3[]){ { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }, 4, 4 );
	b3BodyId hullBody = makeBody( world, b3_dynamicBody );
	b3ShapeId hullShape = b3CreateHullShape( hullBody, &def, initialHull );
	b3Shape_SetHull( hullShape, nextHull );
	const b3HullData* gotHull = b3Shape_GetHull( hullShape );
	const b3Vec3* hullPoints = b3GetHullPoints( gotHull );
	printf( "hull %d %.9g %.9g %.9g", gotHull->vertexCount, gotHull->center.x, gotHull->center.y,
		gotHull->center.z );
	for ( int i = 0; i < gotHull->vertexCount; ++i )
		printf( " %.9g %.9g %.9g", hullPoints[i].x, hullPoints[i].y, hullPoints[i].z );
	printf( "\n" );

	b3MeshData* meshA = makeMesh( 0.0f );
	b3MeshData* meshB = makeMesh( 2.0f );
	b3BodyId meshBody = makeBody( world, b3_staticBody );
	b3ShapeId meshShape = b3CreateMeshShape( meshBody, &def, meshA, (b3Vec3){ 1, 1, 1 } );
	b3Vec3 meshScale = { 2.0f, 0.5f, -1.5f };
	b3Shape_SetMesh( meshShape, meshB, meshScale );
	b3Mesh gotMesh = b3Shape_GetMesh( meshShape );
	const b3Vec3* meshVertices = b3GetMeshVertices( gotMesh.data );
	printf( "mesh %.9g %.9g %.9g %d %d", gotMesh.scale.x, gotMesh.scale.y, gotMesh.scale.z,
		gotMesh.data->vertexCount, gotMesh.data->triangleCount );
	for ( int i = 0; i < gotMesh.data->vertexCount; ++i )
		printf( " %.9g %.9g %.9g", meshVertices[i].x, meshVertices[i].y, meshVertices[i].z );
	printf( "\n" );

	b3SurfaceMaterial meshMaterial = b3DefaultSurfaceMaterial();
	meshMaterial.friction = 0.78f;
	meshMaterial.restitution = 0.19f;
	meshMaterial.rollingResistance = 0.28f;
	meshMaterial.tangentVelocity = ( b3Vec3 ){ -1.0f, 2.0f, -3.0f };
	meshMaterial.userMaterialId = UINT64_C( 0x0fedcba987654321 );
	meshMaterial.customColor = UINT32_C( 0x87654321 );
	b3Shape_SetMeshMaterial( meshShape, meshMaterial, 0 );
	meshMaterial = b3Shape_GetMeshSurfaceMaterial( meshShape, 0 );
	printf( "meshMaterial %.9g %.9g %.9g %.9g %.9g %.9g %llu %u\n", meshMaterial.friction,
		meshMaterial.restitution, meshMaterial.rollingResistance, meshMaterial.tangentVelocity.x,
		meshMaterial.tangentVelocity.y, meshMaterial.tangentVelocity.z,
		(unsigned long long)meshMaterial.userMaterialId, meshMaterial.customColor );

	b3DestroyWorld( world );
	b3DestroyHull( initialHull );
	b3DestroyHull( nextHull );
	b3DestroyMesh( meshA );
	b3DestroyMesh( meshB );
	return 0;
}
