// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
struct b3World;
b3World* b3GetWorldFromId( b3WorldId );
uint64_t b3HashWorldState( b3World* );
}

// Only presentation is inert; the selected classes and AddGroundBox are compiled verbatim.
struct Camera
{
	void SetView( float, float, float, b3Vec3 ) {}
};
namespace ImGui
{
void PushItemWidth( float ) {}
float GetFontSize() { return 14.0f; }
bool SliderFloat( const char*, float*, float, float, const char* ) { return false; }
void PopItemWidth() {}
}
void SetGroundShape( b3ShapeId ) {}

struct SampleContext
{
	Camera camera;
	bool restart = false;
	float hertz = 60.0f;
	int subStepCount = 4;
	int workerCount = 1;
	bool enableSleep = true;
	bool enableWarmStarting = true;
	bool enableContinuous = true;
	float recycleDistance = 0.05f;
};

class Sample
{
public:
	explicit Sample( SampleContext* context ) : m_context( context ), m_camera( &context->camera )
	{
		b3WorldDef worldDef = b3DefaultWorldDef();
		worldDef.workerCount = context->workerCount;
		worldDef.enableSleep = context->enableSleep;
		worldDef.capacity = {};
		m_worldId = b3CreateWorld( &worldDef );
		b3World_SetContactRecycleDistance( m_worldId, context->recycleDistance );
	}
	virtual ~Sample() { b3DestroyWorld( m_worldId ); }
	virtual bool DrawControls() { return false; }
	void Step()
	{
		b3World_EnableSleeping( m_worldId, m_context->enableSleep );
		b3World_EnableWarmStarting( m_worldId, m_context->enableWarmStarting );
		b3World_EnableContinuous( m_worldId, m_context->enableContinuous );
		b3World_Step( m_worldId, 1.0f / m_context->hertz, m_context->subStepCount );
	}
	b3BodyId AddGroundBox( float extent );
	SampleContext* m_context;
	Camera* m_camera;
	b3WorldId m_worldId;
};

// SAMPLE_SOURCES

int main( int argc, char** argv )
{
	if ( argc != 3 || ( strcmp( argv[1], "box_stack" ) && strcmp( argv[1], "bridge" ) ) )
	{
		fprintf( stderr, "usage: sample box_stack|bridge <steps>\n" );
		return 2;
	}
	SampleContext context;
	Sample* sample = strcmp( argv[1], "box_stack" ) == 0 ? BoxStack::Create( &context ) : Bridge::Create( &context );
	for ( int i = 0; i < atoi( argv[2] ); ++i )
	{
		sample->Step();
		printf( "%d 0x%016llx\n", i, (unsigned long long)b3HashWorldState( b3GetWorldFromId( sample->m_worldId ) ) );
	}
	delete sample;
}
