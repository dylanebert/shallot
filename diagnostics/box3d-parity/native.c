// SPDX-License-Identifier: MIT
// Native side of the box3d-parity oracles (divergence.oracle.ts, phases.oracle.ts), linked against Box3D
// 47d7f7cc's box3d and shared libraries. Builds a scene as benchmark/main.c does, steps it at 1/60 with 4
// substeps, and prints the world hash after every step, plus the state the oracles diff or time.
//
//   native <scene> <workers> <steps>
//
// Scenes: joint_grid, rain and junkyard are shared/benchmarks.c's CreateJointGrid, CreateRain/StepRain and
// CreateJunkyard/StepJunkyard at full size. rain-n and junk are copies of the last two with the size as a
// parameter: rain-n takes RAIN_COUNT (grid count, 10 in rain) and RAIN_GROUP (humans per group, 3); junk
// takes ROCKS="X,Y,Z;..." (grid indices of the rocks to keep, all 24 x 21 x 21 in junkyard).
// Environment: COLORS=<step> prints the graph colors' body bits after that step; CACHE=<contact> prints
// that contact's SAT cache after every step; PROBE=<step> FOCUS=<body> calls b3CollideHulls before that
// step on each hull pair touching the body, with a fresh and with the live cache; PROFILE=<step> prints
// b3World_GetProfile's fields, in b3Profile order and in milliseconds, and b3World_GetCounters' contact
// counts after that step and every later one.
#include "benchmarks.h"
#include "human.h"

#include "body.h"
#include "constraint_graph.h"
#include "contact.h"
#include "physics_world.h"
#include "shape.h"

#include "box3d/box3d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t b3HashWorldState( b3World* world );

static uint32_t bits( float f )
{
	uint32_t u;
	memcpy( &u, &f, 4 );
	return u;
}

// --- rain-n: CreateRain/StepRain with RAIN_GRID_COUNT and RAIN_GROUP_SIZE as parameters ---

#define RAIN_GRID_SIZE 15.0f
#define RAIN_MAX_GROUP 3
typedef struct { Human humans[RAIN_MAX_GROUP]; } Group;
static int g_count, g_group, g_columnCount, g_columnIndex;
static Group* g_groups;
static b3MeshData *g_gridMesh, *g_torusMesh;

static void CreateRainN( b3WorldId worldId )
{
	g_count = getenv( "RAIN_COUNT" ) ? atoi( getenv( "RAIN_COUNT" ) ) : 10;
	g_group = getenv( "RAIN_GROUP" ) ? atoi( getenv( "RAIN_GROUP" ) ) : 3;
	g_columnCount = g_columnIndex = 0;
	g_groups = calloc( g_count * g_count, sizeof( Group ) );
	int halfMeshGridRows = 4;
	float meshGridCellWidth = RAIN_GRID_SIZE / ( 2.0f * halfMeshGridRows );
	g_gridMesh = b3CreateGridMesh( 2 * halfMeshGridRows, 2 * halfMeshGridRows, meshGridCellWidth, 1, true );
	g_torusMesh = b3CreateTorusMesh( 16, 16, 0.25f * RAIN_GRID_SIZE, 1.0f );
	float span = RAIN_GRID_SIZE * g_count;
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	bodyDef.position.x = -0.5f * span + 0.5f * RAIN_GRID_SIZE;
	for ( int i = 0; i < g_count; ++i )
	{
		bodyDef.position.z = -0.5f * span + 0.5f * RAIN_GRID_SIZE;
		for ( int j = 0; j < g_count; ++j )
		{
			b3BodyId body = b3CreateBody( worldId, &bodyDef );
			b3CreateMeshShape( body, &shapeDef, g_gridMesh, b3Vec3_one );
			b3CreateMeshShape( body, &shapeDef, g_torusMesh, b3Vec3_one );
			bodyDef.position.z += RAIN_GRID_SIZE;
		}
		bodyDef.position.x += RAIN_GRID_SIZE;
	}
}

static void CreateGroupN( b3WorldId worldId, int rowIndex, int columnIndex )
{
	int groupIndex = rowIndex * g_count + columnIndex;
	float span = g_count * RAIN_GRID_SIZE;
	float groupDistance = 1.0f * span / g_count;
	b3Pos position;
	position.x = -0.5f * span + groupDistance * ( columnIndex + 0.5f );
	position.y = 20.0f;
	position.z = -0.5f * span + groupDistance * ( rowIndex + 0.5f );
	for ( int i = 0; i < g_group; ++i )
	{
		Human* human = g_groups[groupIndex].humans + i;
		CreateHuman( human, worldId, position, 5.0f, 1.0f, 0.7f, groupIndex, NULL, false );
		position.x += 0.75f;
	}
}

static void StepRainN( b3WorldId worldId, int stepCount )
{
	if ( ( stepCount & 0x2F ) != 0 ) return;
	if ( g_columnCount < g_count )
	{
		for ( int i = 0; i < g_count; ++i ) CreateGroupN( worldId, i, g_columnCount );
		g_columnCount = b3MinInt( g_columnCount + 1, g_count );
		return;
	}
	for ( int i = 0; i < g_count; ++i )
	{
		Group* group = g_groups + i * g_count + g_columnIndex;
		for ( int k = 0; k < g_group; ++k ) DestroyHuman( group->humans + k );
		CreateGroupN( worldId, i, g_columnIndex );
	}
	g_columnIndex = g_columnIndex + 1 >= g_count ? 0 : g_columnIndex + 1;
}

// --- junk: CreateJunkyard/StepJunkyard with the rock set as a parameter ---

static b3BodyId g_pusher;
static float g_degrees;

static void CreateJunk( b3WorldId worldId )
{
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.position.y = -1.0f;
	b3BodyId groundId = b3CreateBody( worldId, &groundDef );
	b3ShapeDef groundShape = b3DefaultShapeDef();
	b3BoxHull box = b3MakeBoxHull( 120.0f, 1.0f, 120.0f );
	b3CreateHullShape( groundId, &groundShape, &box.base );
	box = b3MakeOffsetBoxHull( 1.0f, 8.0f, 50.0f, (b3Vec3){ -50.0f, 8.0f, 0.0f } );
	b3CreateHullShape( groundId, &groundShape, &box.base );
	box = b3MakeOffsetBoxHull( 1.0f, 8.0f, 50.0f, (b3Vec3){ 50.0f, 8.0f, 0.0f } );
	b3CreateHullShape( groundId, &groundShape, &box.base );
	box = b3MakeOffsetBoxHull( 50.0f, 8.0f, 1.0f, (b3Vec3){ 0.0f, 8.0f, -50.0f } );
	b3CreateHullShape( groundId, &groundShape, &box.base );
	box = b3MakeOffsetBoxHull( 50.0f, 8.0f, 1.0f, (b3Vec3){ 0.0f, 8.0f, 50.0f } );
	b3CreateHullShape( groundId, &groundShape, &box.base );

	b3HullData* rockHull = b3CreateRock( 1.5f );
	float height = 24.0f;
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	const char* r = getenv( "ROCKS" );
	int X, Y, Z, n;
	while ( r != NULL && sscanf( r, "%d,%d,%d%n", &X, &Y, &Z, &n ) == 3 )
	{
		bodyDef.position.x = -40.0f + 4.0f * X;
		bodyDef.position.y = 4.0f * Y + height + 1.0f;
		bodyDef.position.z = -40.0f + 4.0f * Z;
		b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
		b3CreateHullShape( bodyId, &shapeDef, rockHull );
		r += n;
		r += *r == ';';
	}
	b3DestroyHull( rockHull );

	b3HullData* hull = b3CreateCylinder( 24.0f, 4.0f, 0.0f, 16 );
	b3BodyDef pusherDef = b3DefaultBodyDef();
	pusherDef.type = b3_kinematicBody;
	pusherDef.position = (b3Pos){ 35.0f, 0.0f, 0.0f };
	g_pusher = b3CreateBody( worldId, &pusherDef );
	g_degrees = 0.0f;
	b3ShapeDef pusherShape = b3DefaultShapeDef();
	b3CreateHullShape( g_pusher, &pusherShape, hull );
	b3DestroyHull( hull );
}

static void StepJunk( b3WorldId worldId, int stepCount )
{
	(void)worldId;
	(void)stepCount;
	float timeStep = 1.0f / 60.0f;
	g_degrees += -6.0f * timeStep;
	b3CosSin cs = b3ComputeCosSin( g_degrees * B3_PI / 180.0f );
	b3WorldTransform target = { .p = { 35.0f * cs.cosine, 0.0f, 35.0f * cs.sine }, .q = b3Quat_identity };
	b3Body_SetTargetTransform( g_pusher, target, timeStep, false );
}

// --- state printers ---

static void printColors( b3World* world, int step )
{
	for ( int c = 0; c < B3_OVERFLOW_INDEX; ++c )
	{
		b3BitSet* set = &world->constraintGraph.colors[c].bodySet;
		for ( uint32_t k = 0; k < set->blockCount * 64; ++k )
		{
			if ( set->bits[k / 64] & ( (uint64_t)1 << ( k % 64 ) ) ) printf( "C %d color %d body %u\n", step, c, k );
		}
	}
}

static void printCache( b3World* world, int step, int contactId )
{
	if ( contactId >= world->contacts.count || world->contacts.data[contactId].contactId != contactId ) return;
	b3Contact* c = world->contacts.data + contactId;
	b3SATCache* s = &c->convexContact.cache.satCache;
	printf( "S %d contact %d bodies %d %d manifolds %d cache sep %08x type %d indexA %d indexB %d hit %d\n", step, contactId,
			c->edges[0].bodyId, c->edges[1].bodyId, c->manifoldCount, bits( s->separation ), s->type, s->indexA, s->indexB, s->hit );
}

static void probe( b3World* world, int step, int focus )
{
	for ( int i = 0; i < world->contacts.count; ++i )
	{
		b3Contact* c = world->contacts.data + i;
		if ( c->contactId != i || ( c->edges[0].bodyId != focus && c->edges[1].bodyId != focus ) ) continue;
		b3Shape* shapeA = world->shapes.data + c->shapeIdA;
		b3Shape* shapeB = world->shapes.data + c->shapeIdB;
		if ( shapeA->type != b3_hullShape || shapeB->type != b3_hullShape ) continue;
		b3WorldTransform xfA = b3GetBodySim( world, world->bodies.data + shapeA->bodyId )->transform;
		b3WorldTransform xfB = b3GetBodySim( world, world->bodies.data + shapeB->bodyId )->transform;
		b3Transform x = b3InvMulWorldTransforms( xfA, xfB );
		printf( "X %d contact %d bodies %d %d transformBtoA %08x %08x %08x %08x %08x %08x %08x\n", step, i, shapeA->bodyId,
				shapeB->bodyId, bits( x.p.x ), bits( x.p.y ), bits( x.p.z ), bits( x.q.v.x ), bits( x.q.v.y ), bits( x.q.v.z ),
				bits( x.q.s ) );
		for ( int live = 0; live < 2; ++live )
		{
			b3SATCache cache = live ? c->convexContact.cache.satCache : (b3SATCache){ 0 };
			b3LocalManifoldPoint points[32];
			b3LocalManifold m = { .points = points };
			b3CollideHulls( &m, 32, shapeA->hull, shapeB->hull, x, &cache );
			printf( "H %d contact %d %s points %d normal %08x %08x %08x\n", step, i, live ? "live" : "fresh", m.pointCount,
					bits( m.normal.x ), bits( m.normal.y ), bits( m.normal.z ) );
		}
	}
}

static b3BodyId mergeBodies[2][4];
static void CreateSleepingMerge( b3WorldId worldId )
{
    int sizes[2] = { 3, 4 };
    if ( getenv( "MERGE_SIZES" ) ) sscanf( getenv( "MERGE_SIZES" ), "%d,%d", sizes, sizes + 1 );
    for ( int group = 0; group < 2; ++group )
    {
        for ( int i = 0; i < sizes[group]; ++i )
        {
            b3BodyDef def = b3DefaultBodyDef();
            def.type = b3_dynamicBody;
            def.position.x = group * 10 + i;
            mergeBodies[group][i] = b3CreateBody( worldId, &def );
            b3ShapeDef shapeDef = b3DefaultShapeDef();
            b3Sphere sphere = { .center = { 0, 0, 0 }, .radius = 0.25f };
            b3CreateSphereShape( mergeBodies[group][i], &shapeDef, &sphere );
            if ( i > 0 )
            {
                b3DistanceJointDef joint = b3DefaultDistanceJointDef();
                joint.base.bodyIdA = mergeBodies[group][i - 1];
                joint.base.bodyIdB = mergeBodies[group][i];
                joint.length = 1;
                b3CreateDistanceJoint( worldId, &joint );
            }
        }
        b3Body_SetAwake( mergeBodies[group][0], false );
    }
    b3DistanceJointDef joint = b3DefaultDistanceJointDef();
    joint.base.bodyIdA = mergeBodies[0][1];
    joint.base.bodyIdB = mergeBodies[1][0];
    joint.length = 9;
    b3CreateDistanceJoint( worldId, &joint );
}
static void StepSleepingMerge( b3WorldId worldId, int step )
{
    (void)worldId;
    if ( step == 1 ) b3Body_ApplyLinearImpulseToCenter( mergeBodies[1][1], (b3Vec3){ 1, 2, 3 }, true );
}

typedef struct
{
	const char* name;
	void ( *capacity )( b3Capacity* );
	void ( *create )( b3WorldId );
	void ( *step )( b3WorldId, int );
} Scene;

int main( int argc, char** argv )
{
	Scene scenes[] = {
		{ "rain", GetRainCapacity, CreateRain, StepRain },
		{ "rain-n", NULL, CreateRainN, StepRainN },
		{ "junkyard", GetJunkyardCapacity, CreateJunkyard, StepJunkyard },
		{ "junk", NULL, CreateJunk, StepJunk },
		{ "joint_grid", NULL, CreateJointGrid, NULL },
		{ "many_pyramids", NULL, CreateManyPyramids, NULL },
		{ "large_pyramid", NULL, CreateLargePyramid, NULL },
        { "sleeping_merge", NULL, CreateSleepingMerge, StepSleepingMerge },
	};
	Scene* scene = NULL;
	for ( int i = 0; argc == 4 && i < (int)( sizeof( scenes ) / sizeof( scenes[0] ) ); ++i )
	{
		if ( strcmp( scenes[i].name, argv[1] ) == 0 ) scene = scenes + i;
	}
	if ( scene == NULL )
	{
		fprintf( stderr, "usage: native rain|rain-n|junkyard|junk|joint_grid|many_pyramids|large_pyramid <workers> <steps>\n" );
		return 2;
	}
	int steps = atoi( argv[3] );
	int colors = getenv( "COLORS" ) ? atoi( getenv( "COLORS" ) ) : -1;
	int cache = getenv( "CACHE" ) ? atoi( getenv( "CACHE" ) ) : -1;
	int probeStep = getenv( "PROBE" ) ? atoi( getenv( "PROBE" ) ) : -1;
	int focus = getenv( "FOCUS" ) ? atoi( getenv( "FOCUS" ) ) : -1;
	int profileFrom = getenv( "PROFILE" ) ? atoi( getenv( "PROFILE" ) ) : -1;
	int countersFrom = getenv( "COUNTERS" ) ? atoi( getenv( "COUNTERS" ) ) : profileFrom;

	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.enableContinuous = true;
	worldDef.workerCount = atoi( argv[2] );
	if ( scene->capacity != NULL ) scene->capacity( &worldDef.capacity );
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3World* world = b3GetWorldFromId( worldId );
	scene->create( worldId );
	for ( int i = 0; i < steps; ++i )
	{
		if ( scene->step != NULL ) scene->step( worldId, i );
		if ( i == probeStep ) probe( world, i, focus );
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
		printf( "%d 0x%016llx\n", i, (unsigned long long)b3HashWorldState( world ) );
		if ( profileFrom >= 0 && i >= profileFrom )
		{
			b3Profile p = b3World_GetProfile( worldId );
			const float* field = &p.step;
			printf( "F %d", i );
			for ( int k = 0; k < (int)( sizeof( p ) / sizeof( float ) ); ++k ) printf( " %.4f", field[k] );
			printf( "\n" );
		}
		if ( countersFrom >= 0 && i >= countersFrom )
		{
			b3Counters n = b3World_GetCounters( worldId );
			int manifolds = 0;
			for ( int k = 0; k < B3_CONTACT_MANIFOLD_COUNT_BUCKETS; ++k ) manifolds += n.manifoldCounts[k];
			printf( "N %d contacts %d awake %d manifolds %d recycled %d sat %d satHit %d joints %d\n", i, n.contactCount,
					n.awakeContactCount, manifolds, n.recycledContactCount, n.satCallCount, n.satCacheHitCount, n.jointCount );
		}
		if ( i == colors ) printColors( world, i );
		if ( cache >= 0 ) printCache( world, i, cache );
	}
	b3DestroyWorld( worldId );
	return 0;
}
