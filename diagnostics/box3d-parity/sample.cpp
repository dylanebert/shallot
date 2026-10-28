// SPDX-License-Identifier: MIT
#include "box3d/box3d.h"
#include "utils.h"
#include "benchmarks.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
// SAMPLE_EARCUT
#include <set>
#include <string>

struct Change { int step; std::string name; float value; };
std::vector<Change> changes;
int currentStep = 0;
bool drivingControls = false;

extern "C" {
struct b3World;
b3World* b3GetWorldFromId( b3WorldId );
uint64_t b3HashWorldState( b3World* );
}

// The selected sample classes and AddGroundBox compile verbatim; drawing callbacks below are inert.
struct Camera
{
	// Contact's recording keeps its initial yaw-zero view and reads only the flattened basis.
	bool m_thirdPerson = false;
	b3Pos m_pivot = {};
	void UpdateTransform() {}
	void SetView( float, float, float, b3Vec3 ) {}
	b3Vec3 GetForward() const { return { 0, 0, 1 }; }
	b3Vec3 GetRight() const { return { 1, 0, 0 }; }
};
constexpr int KEY_W = 1, KEY_S = 2, KEY_A = 3, KEY_D = 4, KEY_L = 5, KEY_T = 6;
constexpr int ACTION_PRESS = 1;
void sapp_lock_mouse( bool ) {}
bool IsKeyDown( int key ) {
	float value = 0;
	const char* names[] = { "", "W", "S", "A", "D", "L" };
	for ( const Change& change : changes )
		if ( change.step <= currentStep && change.name == names[key] ) value = change.value;
	return value != 0;
}
void DrawGroundGrid( int ) {}
template<class T> void DrawAxes( T, float ) {}
int MakeColor( int color ) { return color; }
void DrawPoint( b3Pos, float, int ) {}
void DrawLine( b3Pos, b3Pos, int ) {}
void DrawWireSphere( b3WorldTransform, const b3Sphere*, int, int ) {}
void DrawString3D( b3Pos, int, const char*, ... ) {}
int MakeColorAlpha( int color, float ) { return color; }
constexpr int TRANSPARENT_SHADOW_NONE = 0;
void DrawSolidSphere( b3WorldTransform, b3Sphere, int ) {}
void DrawSphereEx( b3WorldTransform, float, int, float, float, int ) {}
void DrawHull( b3WorldTransform, const b3HullData*, int ) {}
bool recordQueriesAndBuilds = false;
void PrintFloat( float value ) { unsigned bits; memcpy( &bits, &value, 4 ); printf( " %08x", bits ); }
void PrintCast( const b3RayResult& r ) {
    printf( "cast %d", r.hit );
    for ( float v : { float(r.point.x), float(r.point.y), float(r.point.z), r.normal.x, r.normal.y, r.normal.z, r.fraction } ) PrintFloat( v );
    printf( "\n" );
}
b3RayResult RecordedRay( b3WorldId world, b3Pos origin, b3Vec3 translation, b3QueryFilter filter ) {
    b3RayResult r = b3World_CastRayClosest( world, origin, translation, filter );
    if ( recordQueriesAndBuilds ) PrintCast( r );
    return r;
}
void RecordedShapeCast( b3WorldId world, b3Pos origin, const b3ShapeProxy* proxy, b3Vec3 translation, b3QueryFilter filter, b3CastResultFcn* callback, void* context ) {
    struct Context { b3CastResultFcn* callback; void* user; b3RayResult result; } c = { callback, context, {} };
    b3World_CastShape( world, origin, proxy, translation, filter,
        []( b3ShapeId id, b3Pos point, b3Vec3 normal, float fraction, uint64_t material, int triangle, int child, void* ptr ) {
            Context* c = (Context*)ptr;
            c->result.point = point; c->result.normal = normal; c->result.fraction = fraction; c->result.hit = true;
            return c->callback( id, point, normal, fraction, material, triangle, child, c->user );
        }, &c );
    if ( recordQueriesAndBuilds ) PrintCast( c.result );
}
void PrintHull( const b3HullData* hull ) {
    if ( !recordQueriesAndBuilds ) return;
    printf( "hull" ); PrintFloat( hull->surfaceArea );
    const b3Vec3* points = (const b3Vec3*)((const char*)hull + hull->pointOffset);
    for ( int i = 0; i < hull->vertexCount; ++i ) { PrintFloat( points[i].x ); PrintFloat( points[i].y ); PrintFloat( points[i].z ); }
    printf( "\n" );
}
b3HullData* RecordedHull( const b3Vec3* points, int count, int maxCount ) {
    b3HullData* hull = b3CreateHull( points, count, maxCount ); PrintHull( hull ); return hull;
}
b3HullData* RecordedClone( const b3HullData* hull, b3Transform transform, b3Vec3 scale ) {
    b3HullData* result = b3CloneAndTransformHull( hull, transform, scale ); PrintHull( result ); return result;
}
namespace ImGui
{
void Text( const char*, ... ) {}
void PushItemWidth( float ) {}
float GetFontSize() { return 14.0f; }
bool SliderFloat( const char* name, float* value, float, float, const char* = "%.3f" ) {
    std::string label(name);
    const auto suffix = label.find("##");
    if (suffix != std::string::npos) {
        const auto group = label.substr(suffix + 2);
        if (group == "Door") label.resize(suffix);
        else if (drivingControls && (group == "Suspension" || group == "Steering"))
            label = group + " " + label.substr(0, suffix);
    }
    name = label.c_str();
    for ( const Change& change : changes )
        if ( change.step == currentStep && change.name == name ) { *value = change.value; return true; }
    return false;
}
bool SliderInt( const char* name, int* value, int, int ) {
    float result = float(*value);
    bool changed = SliderFloat( name, &result, 0, 0, "" );
    *value = int(result);
    return changed;
}
bool SliderFloat3( const char* name, float* value, float min, float max, const char* format ) {
    bool changed = false;
    for ( int i = 0; i < 3; ++i ) {
        std::string label = std::string(name) + " " + "XYZ"[i];
        changed = SliderFloat( label.c_str(), value + i, min, max, format ) || changed;
    }
    return changed;
}
bool RadioButton( const char* name, int* value, int selected ) {
    float pressed = 0;
    if ( SliderFloat( name, &pressed, 0, 1, "" ) && pressed != 0 ) { *value = selected; return true; }
    return false;
}
bool RadioButton( const char* name, bool ) {
    float pressed = 0;
    return SliderFloat( name, &pressed, 0, 1, "" ) && pressed != 0;
}
bool Checkbox( const char* name, bool* value ) {
    float result = *value ? 1.0f : 0.0f;
    bool changed = SliderFloat( name, &result, 0, 1, "" );
    if ( changed ) *value = result != 0;
    return changed;
}
bool Button( const char* name ) {
    float value = 0;
    return SliderFloat( name, &value, 0, 1, "" ) && value != 0;
}
void Separator() {}
void PopItemWidth() {}
}
void SetGroundShape( b3ShapeId ) {}
struct GuiDraw { bool drawJoints = true; float forceScale = 1.0f; };
GuiDraw* GetGuiDraw() { static GuiDraw draw; return &draw; }

struct SampleContext
{
	Camera camera;
	bool restart = false;
	bool pause = false;
	int singleStep = 0;
	float hertz = 60.0f;
	int subStepCount = 4;
	int workerCount = 1;
	bool enableSleep = true;
	bool enableWarmStarting = true;
	bool enableContinuous = true;
	float recycleDistance = 0.05f;
};

// SAMPLE_EVENTS

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
	void ResetProfile() { m_stepCount = 0; }
	void CreateWorld( const b3Capacity* capacity ) {
		b3DestroyWorld( m_worldId );
		b3WorldDef def = b3DefaultWorldDef();
		def.workerCount = m_context->workerCount;
		def.enableSleep = m_context->enableSleep;
		def.capacity = capacity ? *capacity : b3Capacity{};
		m_worldId = b3CreateWorld( &def );
		b3World_SetContactRecycleDistance( m_worldId, m_context->recycleDistance );
	}
	virtual void Keyboard( int, int, int ) {}
	void ToggleThirdPerson() { m_camera->m_thirdPerson = !m_camera->m_thirdPerson; }
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
		m_stepHash = b3HashWorldState( b3GetWorldFromId( m_worldId ) );
        if ( !recordQueriesAndBuilds ) {
		    printf( "%016llx", (unsigned long long)m_stepHash );
		    if ( recordEvents ) PrintEvents( m_worldId );
		    printf( "\n" );
        }
		++m_stepCount;
	}
	bool m_stepWhilePaused = true;
	int m_stepCount = 0;
	static constexpr bool m_isDebug = false;
	float m_shadowSplitFar = 0;
	uint64_t m_stepHash = 0;
	b3BodyId AddGroundBox( float extent );
	SampleContext* m_context;
	Camera* m_camera;
	b3WorldId m_worldId;
};

// The sample translation unit uses JavaScript's gameplay math environment: double libm,
// rounded to float. Box3D's separately compiled library and its kernel math are untouched.
#define sinf(value) static_cast<float>(std::sin(static_cast<double>(value)))
#define cosf(value) static_cast<float>(std::cos(static_cast<double>(value)))

#define b3World_CastRayClosest RecordedRay
#define b3World_CastShape RecordedShapeCast
#define b3CreateHull RecordedHull
#define b3CloneAndTransformHull RecordedClone
// SAMPLE_SOURCES

int main( int argc, char** argv )
{
	if ( argc != 3 && argc != 4 )
	{
		fprintf( stderr, "usage: sample <scene> <steps> [grab|control-sequence-file]\n" );
		return 2;
	}
	recordEvents = !strcmp( argv[1], "benchmarkSensor" ) || !strcmp( argv[1], "joint" ) || !strcmp( argv[1], "sensor" ) || !strcmp( argv[1], "contact" ) || !strcmp( argv[1], "hit" ) || !strcmp( argv[1], "motor" );
	b3SetLogFcn( []( const char* message ) { fprintf( stderr, "%s\n", message ); } );
    recordQueriesAndBuilds = !strcmp( argv[1], "heightField" ) || !strcmp( argv[1], "hull" );
	SampleContext context;
	struct Entry { const char* name; Sample* (*create)(SampleContext*); };
	const Entry entries[] = {
        { "bodyType", BodyType::Create }, { "weeble", Weeble::Create }, { "disable", DisableBody::Create },
        { "spinningBook", SpinningBooks::Create }, { "gyroscopicTorque", GyroscopicTorque::Create },
        { "kinematic", Kinematic::Create }, { "lockMixing", LockMixing::Create },
        { "fixedRotation", FixedRotation::Create }, { "offsetKinematic", OffsetKinematic::Create },
		{ "largeWorld", BenchmarkLargeWorld::Create }, { "largePyramid", BenchmarkLargePyramid::Create }, { "widePyramid", BenchmarkWidePyramid::Create },
		{ "manyPyramids", BenchmarkManyPyramids::Create }, { "jointGrid", BenchmarkJointGrid::Create },
		{ "fallingBoxes", FallingBoxes::Create },
		{ "rain", BenchmarkRain::Create },
		{ "junkyard", BenchmarkJunkyard::Create },
		{ "washer", BenchmarkWasher::Create },
		{ "convexPile", BenchmarkConvexPile::Create },
		{ "candyCups", CandyCups::Create },
		{ "fallingTrees", BenchmarkFallingTrees::Create },
		{ "chains", BenchmarkChains::Create },
        { "benchmarkSensor", BenchmarkSensor::Create },
		{ "explosion", BenchmarkExplosion::Create },
		{ "destruction", BenchmarkDestruction::Create },
        { "heightField", BenchmarkHeightField::Create },
        { "hull", BenchmarkHull::Create },
		{ "cardHouse", CardHouse::Create }, { "sphereStack", SphereStack::Create },
		{ "jenga", JengaStack::Create },
		{ "capsuleStack", CapsuleStack::Create }, { "singleBox", SingleBox::Create },
		{ "cylinder", Cylinder::Create }, { "cylinderStack", CylinderStack::Create },
		{ "dominoes", Dominoes::Create }, { "wedge", Wedge::Create },
		{ "arch", Arch::Create }, { "doubleDomino", DoubleDomino::Create },
		{ "pyramid2D", Pyramid2D::Create }, { "edgeCrossing", EdgeCrossing::Create },
		{ "box_stack", BoxStack::Create }, { "bridge", Bridge::Create },
		{ "driving", Driving::Create }, { "gearLift", GearLift::Create }, { "door", Door::Create }, { "distance", DistanceJoint::Create }, { "spherical", SphericalJoint::Create }, { "motionLocks", MotionLocks::Create },
        { "ballAndChain", BallAndChain::Create }, { "foldOut", FoldOut::Create }, { "topDownFriction", TopDownFriction::Create }, { "weld", WeldJoint::Create },
		{ "revolute", RevoluteJoint::Create }, { "prismatic", PrismaticJoint::Create },
		{ "wheel", WheelJoint::Create }, { "filter", FilterJoint::Create }, { "parallel", ParallelJoint::Create },
		{ "motor", MotorJoint::Create }, { "joint", JointEvent::Create },
		{ "sensor", SensorVisit::Create }, { "contact", ContactEvent::Create }, { "hit", HitEvent::Create },
	};
	Sample* sample = nullptr;
	for ( const Entry& entry : entries ) if ( strcmp( argv[1], entry.name ) == 0 ) sample = entry.create( &context );
	if ( !sample ) { fprintf( stderr, "unknown sample: %s\n", argv[1] ); return 2; }
	drivingControls = !strcmp( argv[1], "driving" );
	if ( drivingControls ) context.camera.m_thirdPerson = true;
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
		for ( const Change& change : changes ) if ( change.step == i ) {
			if ( change.name == "grabBody" ) {
				const int tag = (int)change.value;
				assert( tag > 0 && tag <= (int)recordedBodies.size() );
				b3Pos p = b3Body_GetPosition( recordedBodies[tag - 1] );
				sample->Grab( { { p.x, p.y, p.z + 10 }, { 0, 0, -20 } } );
			} else if ( change.name == "grabY" ) sample->m_mousePoint.y = change.value;
			else if ( change.name == "release" ) sample->Release();
		}
		currentStep = i;
		sample->DrawControls();
		sample->Step();
        if ( recordQueriesAndBuilds ) sample->Render();
	}
	delete sample;
}
