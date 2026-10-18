// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

struct Change { int step; std::string name; float value; };
std::vector<Change> changes;
int currentStep = 0;

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
bool SliderFloat( const char* name, float* value, float, float, const char* ) {
    for ( const Change& change : changes )
        if ( change.step == currentStep && change.name == name ) { *value = change.value; return true; }
    return false;
}
bool Checkbox( const char* name, bool* value ) {
    float result = *value ? 1.0f : 0.0f;
    bool changed = SliderFloat( name, &result, 0, 1, "" );
    if ( changed ) *value = result != 0;
    return changed;
}
void Separator() {}
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
	virtual void Render() {}
	void DrawTextLine( const char*, ... ) {}
	struct PickRay { b3Pos origin; b3Vec3 translation; };
	b3BodyId m_mouseBodyId = {};
	b3JointId m_mouseJointId = {};
	b3Pos m_mousePoint = {};
	float m_mouseForceScale = 100.0f;
	float m_mouseFraction = 0.0f;
	void SetSelectedBody( b3BodyId ) {}
	void Grab( PickRay pickRay ) {
// SAMPLE_GRAB
	}
	void Release()
// SAMPLE_UP
	virtual void Step()
	{
		if ( B3_IS_NON_NULL( m_mouseBodyId ) )
			b3Body_SetTargetTransform( m_mouseBodyId, { m_mousePoint, b3Quat_identity }, 1.0f / m_context->hertz, true );
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
	if ( argc != 3 && argc != 4 )
	{
		fprintf( stderr, "usage: sample <scene> <steps> [grab|control-sequence-file]\n" );
		return 2;
	}
	SampleContext context;
	struct Entry { const char* name; Sample* (*create)(SampleContext*); };
	const Entry entries[] = {
		{ "box_stack", BoxStack::Create }, { "bridge", Bridge::Create },
		{ "revolute", RevoluteJoint::Create }, { "prismatic", PrismaticJoint::Create },
		{ "wheel", WheelJoint::Create }, { "filter", FilterJoint::Create }, { "parallel", ParallelJoint::Create },
	};
	Sample* sample = nullptr;
	for ( const Entry& entry : entries ) if ( strcmp( argv[1], entry.name ) == 0 ) sample = entry.create( &context );
	if ( !sample ) { fprintf( stderr, "unknown sample: %s\n", argv[1] ); return 2; }
	bool grab = argc == 4 && strcmp( argv[3], "grab" ) == 0;
	if ( argc == 4 && !grab ) {
		FILE* file = fopen( argv[3], "r" );
		if ( !file ) { perror( argv[3] ); delete sample; return 2; }
		char line[256], name[128]; int step; float value;
		while ( fgets( line, sizeof(line), file ) ) {
			if ( sscanf( line, "%d\t%127[^\t]\t%f", &step, name, &value ) != 3 ) { fclose(file); delete sample; return 2; }
			changes.push_back( { step, name, value } );
		}
		fclose( file );
	}
	for ( int i = 0; i < atoi( argv[2] ); ++i )
	{
		if ( grab )
		{
			if ( i == 1 ) sample->Grab( { { 0, 25.75f, 10 }, { 0, 0, -20 } } );
			if ( i > 1 && i < 31 ) sample->m_mousePoint.x = 0.05f * ( i - 1 );
			if ( i == 31 ) sample->Release();
		}
		currentStep = i;
		sample->DrawControls();
		sample->Step();
		printf( "%d 0x%016llx\n", i, (unsigned long long)b3HashWorldState( b3GetWorldFromId( sample->m_worldId ) ) );
	}
	delete sample;
}
