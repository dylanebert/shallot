// SPDX-License-Identifier: MIT
// Native body-definition hashes and getter snapshots for body-definitions.oracle.ts.
#include "physics_world.h"
#include "box3d/box3d.h"

#include <stdio.h>
#include <string.h>

uint64_t b3HashWorldState( b3World* world );

static b3BodyId snapshotBody;

static uint64_t hash( b3WorldId id )
{
	b3World* world = b3GetWorldFromId( id );
	return b3HashWorldState( world );
}

static b3WorldId createWorld( void )
{
	b3WorldDef definition = b3DefaultWorldDef();
	definition.workerCount = 1;
	return b3CreateWorld( &definition );
}

static void printFloatBits( float value )
{
	uint32_t bits;
	memcpy( &bits, &value, sizeof( bits ) );
	printf( " %08x", bits );
}

static void printHash( b3WorldId id )
{
	b3MotionLocks locks = b3Body_GetMotionLocks( snapshotBody );
	b3MassData massData = b3Body_GetMassData( snapshotBody );
	uint32_t lockBits = ( locks.linearX ? 1u : 0u ) | ( locks.linearY ? 2u : 0u ) |
		( locks.linearZ ? 4u : 0u ) | ( locks.angularX ? 8u : 0u ) |
		( locks.angularY ? 16u : 0u ) | ( locks.angularZ ? 32u : 0u );

	printf( "%016llx %d", (unsigned long long)hash( id ), (int)b3Body_GetType( snapshotBody ) );
	printFloatBits( b3Body_GetLinearDamping( snapshotBody ) );
	printFloatBits( b3Body_GetAngularDamping( snapshotBody ) );
	printFloatBits( b3Body_GetGravityScale( snapshotBody ) );
	printFloatBits( b3Body_GetSleepThreshold( snapshotBody ) );
	printf( " %d %d %d %u %d %d %d", b3Body_IsSleepEnabled( snapshotBody ),
		b3Body_IsEnabled( snapshotBody ), b3Body_IsBullet( snapshotBody ), lockBits,
		b3Body_IsFastRotationAllowed( snapshotBody ),
		b3Body_IsContactRecyclingEnabled( snapshotBody ), b3Body_IsAwake( snapshotBody ) );
	printFloatBits( massData.mass );
	printFloatBits( massData.center.x );
	printFloatBits( massData.center.y );
	printFloatBits( massData.center.z );
	printFloatBits( massData.inertia.cx.x );
	printFloatBits( massData.inertia.cx.y );
	printFloatBits( massData.inertia.cx.z );
	printFloatBits( massData.inertia.cy.x );
	printFloatBits( massData.inertia.cy.y );
	printFloatBits( massData.inertia.cy.z );
	printFloatBits( massData.inertia.cz.x );
	printFloatBits( massData.inertia.cz.y );
	printFloatBits( massData.inertia.cz.z );
	putchar( '\n' );
}

static int printContactHash( b3WorldId id )
{
	if ( b3World_GetCounters( id ).contactCount == 0 )
	{
		fprintf( stderr, "contact-recycling oracle did not create a contact\n" );
		return 0;
	}
	printHash( id );
	return 1;
}

int main( void )
{
	b3WorldId world = createWorld();
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId body = b3CreateBody( world, &bodyDef );
	snapshotBody = body;
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	shapeDef.baseMaterial.friction = 0.5f;
	b3BoxHull box = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
	b3CreateHullShape( body, &shapeDef, &box.base );
	printHash( world );
	b3DestroyWorld( world );

	world = createWorld();
	bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 1.0f, 4.0f, -2.0f };
	bodyDef.linearVelocity = (b3Vec3){ 0.3f, 0.1f, -0.2f };
	bodyDef.angularVelocity = (b3Vec3){ 0.1f, 0.2f, -0.3f };
	body = b3CreateBody( world, &bodyDef );
	snapshotBody = body;
	shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	shapeDef.baseMaterial.friction = 0.5f;
	b3Sphere sphere = { .center = { 0.25f, 0.1f, -0.15f }, .radius = 0.5f };
	b3CreateSphereShape( body, &shapeDef, &sphere );
	printHash( world );

	b3Body_SetLinearDamping( body, 0.23f );
	printHash( world );
	b3Body_SetAngularDamping( body, 0.31f );
	printHash( world );
	b3Body_SetGravityScale( body, 0.4f );
	printHash( world );
	b3Body_SetSleepThreshold( body, 0.12f );
	printHash( world );

	b3MotionLocks locks = { .linearX = true, .angularZ = true };
	b3Body_SetMotionLocks( body, locks );
	printHash( world );
	locks.angularX = true;
	locks.angularY = true;
	b3Body_SetMotionLocks( body, locks );
	printHash( world );
	locks.angularX = false;
	locks.angularY = false;
	locks.angularZ = false;
	b3Body_SetMotionLocks( body, locks );
	printHash( world );

	b3Body_EnableSleep( body, false );
	printHash( world );
	b3Body_EnableSleep( body, true );
	printHash( world );
	b3Body_SetBullet( body, true );
	printHash( world );
	b3Body_AllowFastRotation( body, true );
	printHash( world );
	b3Body_SetLinearVelocity( body, (b3Vec3){ 0.0f, 0.0f, 0.0f } );
	printHash( world );
	b3Body_SetAngularVelocity( body, (b3Vec3){ 0.0f, 0.0f, 0.0f } );
	printHash( world );
	b3Body_SetAwake( body, false );
	printHash( world );
	b3Body_SetAwake( body, true );
	printHash( world );
	b3Body_SetType( body, b3_kinematicBody );
	printHash( world );
	b3Body_SetType( body, b3_dynamicBody );
	printHash( world );
	b3Body_Disable( body );
	printHash( world );
	b3Body_Enable( body );
	printHash( world );
	b3DestroyWorld( world );

	world = createWorld();
	b3BodyDef floorDef = b3DefaultBodyDef();
	floorDef.position = (b3Pos){ 0.0f, -0.5f, 0.0f };
	b3BodyId floor = b3CreateBody( world, &floorDef );
	shapeDef = b3DefaultShapeDef();
	b3BoxHull floorBox = b3MakeBoxHull( 5.0f, 0.5f, 5.0f );
	b3CreateHullShape( floor, &shapeDef, &floorBox.base );

	bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 0.0f, 3.0f, 0.0f };
	body = b3CreateBody( world, &bodyDef );
	snapshotBody = body;
	shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	b3CreateHullShape( body, &shapeDef, &box.base );
	b3Body_EnableContactRecycling( body, false );
	printHash( world );
	b3Body_SetTransform( body, (b3Pos){ 0.0f, 0.4f, 0.0f }, b3Quat_identity );
	b3World_Step( world, 1.0f / 60.0f, 4 );
	if ( !printContactHash( world ) ) return 2;
	b3Body_EnableContactRecycling( body, true );
	printHash( world );
	b3Body_SetTransform( body, (b3Pos){ 0.0f, 3.0f, 0.0f }, b3Quat_identity );
	b3World_Step( world, 1.0f / 60.0f, 4 );
	b3Body_SetTransform( body, (b3Pos){ 0.0f, 0.4f, 0.0f }, b3Quat_identity );
	b3World_Step( world, 1.0f / 60.0f, 4 );
	if ( !printContactHash( world ) ) return 2;
	b3DestroyWorld( world );
	return 0;
}
