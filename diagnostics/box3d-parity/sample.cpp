// SPDX-License-Identifier: MIT
#define NDEBUG
#include "box3d/box3d.h"
#undef NDEBUG
#include "utils.h"
#include "benchmarks.h"
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <vector>
#include <array>
// SAMPLE_EARCUT
// SAMPLE_MESH_LOADER
#include <set>
#include <string>

struct Change { int step; std::string name; float value; };
std::vector<Change> changes;
int currentStep = 0;
bool drivingControls = false;
bool meshDropRecording = false;
uint64_t SampleTicks() { static uint64_t seed = 12345; return seed++; }

extern "C" {
struct b3World;
b3World* b3GetWorldFromId( b3WorldId );
uint64_t b3HashWorldState( b3World* );
}

void RecordedStep(b3WorldId world, float dt, int substeps) {
    b3World_Step(world, dt, substeps);
    if (meshDropRecording) printf("%016llx\n", (unsigned long long)b3HashWorldState(b3GetWorldFromId(world)));
}

// The selected sample classes and AddGroundBox compile verbatim; drawing callbacks below are inert.
struct PickRay { b3Pos origin; b3Vec3 translation; };
struct Camera
{
	// Contact's recording keeps its initial yaw-zero view and reads only the flattened basis.
	bool m_thirdPerson = false;
	b3Pos m_pivot = {};
    b3Pos m_worldEye = {};
    float m_radius = 5;
    PickRay pickRay = {};
	PickRay BuildPickRay( float, float ) { return pickRay; }
	void UpdateTransform() {}
	void SetView( float, float, float, b3Vec3 ) {}
	b3Vec3 GetForward() const { return { 0, 0, 1 }; }
	b3Vec3 GetRight() const { return { 1, 0, 0 }; }
    b3Pos DrawOrigin() const { return {}; }
};
constexpr int KEY_W = 1, KEY_S = 2, KEY_A = 3, KEY_D = 4, KEY_L = 5, KEY_T = 6, KEY_B = 7;
constexpr int KEY_SPACE = 8, KEY_LEFT_SHIFT = 9, KEY_V = 10;
constexpr int ACTION_PRESS = 1, MOD_SHIFT = 1, MOD_ALT = 2;
void sapp_lock_mouse( bool ) {}
bool IsKeyDown( int key ) {
	float value = 0;
	const char* names[] = { "", "W", "S", "A", "D", "L", "T", "B", "Space", "Shift" };
	for ( const Change& change : changes )
		if ( change.step <= currentStep && change.name == names[key] ) value = change.value;
	return value != 0;
}
void DrawGroundGrid( int ) {}
void SetDrawOrigin( b3Pos ) {}
void DrawCube( b3WorldTransform, b3Vec3, int ) {}
template<class T> void DrawAxes( T, float ) {}
void DrawAxes( b3WorldTransform, float ) {}
using Vec4 = int;
int MakeColor( int color ) { return color; }
FILE* contactDrawing = nullptr;
struct ContactMark { b3Pos point; b3Pos end; std::string label; };
std::vector<ContactMark> contactMarks;
FILE* characterDrawing = nullptr;
std::vector<std::vector<double>> characterMarks;
void DrawPoint( b3Pos p, float size, int ) { if (characterDrawing) characterMarks.push_back({0,p.x,p.y,p.z,size}); }
void DrawLine( b3Pos a, b3Pos b, int ) {
    if (contactDrawing) contactMarks.push_back({a,b,{}});
    if (characterDrawing) characterMarks.push_back({1,a.x,a.y,a.z,b.x,b.y,b.z});
}
void DrawArrow( b3Pos a, b3Pos b, int ) { if (characterDrawing) characterMarks.push_back({2,a.x,a.y,a.z,b.x,b.y,b.z}); }
void DrawWireSphere( b3WorldTransform, const b3Sphere*, int, int ) {}
void DrawString3D( b3Pos, int, const char* format, ... ) {
    if (!contactDrawing) return;
    char text[128]; va_list args; va_start(args, format); vsnprintf(text, sizeof(text), format, args); va_end(args);
    contactMarks.back().label = text;
}
int MakeColorAlpha( int color, float ) { return color; }
constexpr int TRANSPARENT_SHADOW_NONE = 0;
void DrawSolidSphere( b3WorldTransform, b3Sphere, int ) {}
void DrawSphereEx( b3WorldTransform, float, int, float, float, int ) {}
void DrawHull( b3WorldTransform, const b3HullData*, int ) {}
void DrawSolidCapsule( b3WorldTransform t, b3Capsule c, int ) {
    if (characterDrawing) characterMarks.push_back({3,t.p.x,t.p.y,t.p.z,t.q.v.x,t.q.v.y,t.q.v.z,t.q.s,c.center1.x,c.center1.y,c.center1.z,c.center2.x,c.center2.y,c.center2.z,c.radius});
}
void DrawPlane( b3Vec3, b3Pos, int ) {}
FILE* bodyQueries = nullptr;
void BodyQueryFloat(float value) { unsigned bits; memcpy(&bits, &value, 4); fprintf(bodyQueries, " %08x", bits); }
void BodyQueryCast(const b3BodyCastResult& r) {
    if (!bodyQueries) return;
    fprintf(bodyQueries, "cast %d", r.hit);
    for (float v : {float(r.point.x), float(r.point.y), float(r.point.z), r.normal.x, r.normal.y, r.normal.z, r.fraction}) BodyQueryFloat(v);
    fprintf(bodyQueries, " %d %d %llu\n", r.triangleIndex, 0, (unsigned long long)r.userMaterialId);
}
b3BodyCastResult RecordedBodyRay(b3BodyId body, b3Pos origin, b3Vec3 translation, b3QueryFilter filter, float fraction, b3WorldTransform transform) {
    auto r = b3Body_CastRay(body, origin, translation, filter, fraction, transform); BodyQueryCast(r); return r;
}
b3BodyCastResult RecordedBodyShape(b3BodyId body, b3Pos origin, const b3ShapeProxy* proxy, b3Vec3 translation, b3QueryFilter filter, float fraction, bool encroach, b3WorldTransform transform) {
    auto r = b3Body_CastShape(body, origin, proxy, translation, filter, fraction, encroach, transform); BodyQueryCast(r); return r;
}
bool RecordedBodyOverlap(b3BodyId body, b3Pos origin, const b3ShapeProxy* proxy, b3QueryFilter filter, b3WorldTransform transform) {
    bool r = b3Body_OverlapShape(body, origin, proxy, filter, transform);
    if (bodyQueries) fprintf(bodyQueries, "overlap %d\n", r); return r;
}
int RecordedBodyMover(b3BodyId body, b3BodyPlaneResult* planes, int capacity, b3Pos origin, const b3Capsule* capsule, b3QueryFilter filter, b3WorldTransform transform) {
    int count = b3Body_CollideMover(body, planes, capacity, origin, capsule, filter, transform);
    if (bodyQueries) {
        fprintf(bodyQueries, "mover %d\n", count);
        for (int i = 0; i < count; ++i) {
            const auto& p = planes[i].result;
            fprintf(bodyQueries, "plane");
            for (float v : {p.plane.normal.x, p.plane.normal.y, p.plane.normal.z, p.plane.offset, p.point.x, p.point.y, p.point.z}) BodyQueryFloat(v);
            fprintf(bodyQueries, " %d %d %d\n", p.triangleIndex, p.childIndex, p.materialIndex);
        }
    }
    return count;
}
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
#define IM_ARRAYSIZE(_ARR) ((int)(sizeof(_ARR) / sizeof(*(_ARR))))
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
bool SliderInt( const char* name, int* value, int, int, const char* = "%d" ) {
    float result = float(*value);
    bool changed = SliderFloat( name, &result, 0, 0, "" );
    *value = int(result);
    return changed;
}
bool Combo( const char* name, int* value, const char* const*, int ) {
    return SliderInt( name, value, 0, 0 );
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
    float mouseX = 0, mouseY = 0;
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
    float m_launchSpeedScale = 1.0f;
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
	virtual void MouseDown( b3Vec2, int, int ) {}
	virtual void MouseUp( b3Vec2, int ) {}
	virtual void MouseMove( b3Vec2 ) {}
	virtual void Keyboard( int, int, int ) {}
	void ToggleThirdPerson() { m_camera->m_thirdPerson = !m_camera->m_thirdPerson; }
	virtual bool DrawControls() { return false; }
	virtual void Render() {}
	void DrawTextLine( const char*, ... ) {}

    b3Vec2 m_mouseDelta = {};
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
        if (m_context->pause) return;
		RecordedStep( m_worldId, 1.0f / m_context->hertz, m_context->subStepCount );
		m_stepHash = b3HashWorldState( b3GetWorldFromId( m_worldId ) );
        if ( !recordQueriesAndBuilds && !meshDropRecording ) {
		    printf( "%016llx", (unsigned long long)m_stepHash );
		    if ( recordEvents ) PrintEvents( m_worldId );
		    printf( "\n" );
        }
		++m_stepCount;
	}
	bool m_stepWhilePaused = true;
    bool m_didStep = true;
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
#define b3Body_CastRay RecordedBodyRay
#define b3Body_CastShape RecordedBodyShape
#define b3Body_OverlapShape RecordedBodyOverlap
#define b3Body_CollideMover RecordedBodyMover
#define b3GetTicks SampleTicks
#define b3World_Step RecordedStep
// SAMPLE_SOURCES

int main( int argc, char** argv )
{
	if ( argc != 3 && argc != 4 )
	{
		fprintf( stderr, "usage: sample <scene> <steps> [grab|control-sequence-file]\n" );
		return 2;
	}
	recordEvents = !strcmp( argv[1], "benchmarkSensor" ) || !strcmp( argv[1], "joint" ) || !strcmp( argv[1], "sensor" ) || !strcmp( argv[1], "contact" ) || !strcmp( argv[1], "hit" ) || !strcmp( argv[1], "motor" ) || !strcmp( argv[1], "move" ) || !strcmp( argv[1], "persistentContact" ) || !strcmp( argv[1], "sensorHits" );
	b3SetLogFcn( []( const char* message ) { fprintf( stderr, "%s\n", message ); } );
    recordQueriesAndBuilds = !strcmp( argv[1], "heightField" ) || !strcmp( argv[1], "hull" );
    meshDropRecording = !strcmp(argv[1], "meshDrop");
	SampleContext context;
    if (getenv("SAMPLE_DISABLE_SLEEP")) context.enableSleep = false;
	struct Entry { const char* name; Sample* (*create)(SampleContext*); };
	const Entry entries[] = {
        { "simpleCompound", SimpleCompound::Create }, { "compoundSpheres", CompoundSpheres::Create },
        { "compoundHulls", CompoundHulls::Create }, { "tileFloor", TileFloor::Create }, { "meshTile", MeshTile::Create },
        { "bulletVersusStack", BulletVersusStack::Create }, { "meshDrop", MeshDrop::Create },
        { "thinWall", ThinWall::Create }, { "bounceHouse", BounceHouse::Create },
        { "spinningStick", SpinningStick::Create }, { "needleMesh", NeedleMesh::Create },
        { "humpMesh", HumpMesh::Create }, { "isFast", IsFast::Create }, { "stall", Stall::Create },
        { "inclinedPlane", InclinedPlane::Create }, { "rollingResistance", RollingResistance::Create },
        { "highResistance", HighResistance::Create }, { "isotropicFriction", IsotropicFriction::Create },
        { "slideTwist", SlideTwist::Create }, { "conveyorBelt", ConveyorBelt::Create },
        { "restitution", Restitution::Create }, { "staticInvoke", StaticInvoke::Create }, { "wind", Wind::Create },
        { "conveyorMesh", ConveyorMesh::Create }, { "windDrop", WindDrop::Create }, { "windFlap", WindFlap::Create },
        { "rigidBody", RigidBodyCharacter::Create }, { "geometricMover", GeometricMover::Create }, { "capsulePlane", CapsulePlane::Create }, { "moverOverlap", MoverOverlap::Create },
        { "classRing", ClassRing::Create }, { "cast", BodyCast::Create },
        { "bodyType", BodyType::Create }, { "weeble", Weeble::Create }, { "disable", DisableBody::Create },
        { "gyroscopicPrecession", GyroscopicPrecession::Create },
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
        { "move", MoveEvent::Create }, { "persistentContact", PersistentContact::Create },
        { "sensorHits", SensorHits::Create },
	};
	Sample* sample = nullptr;
	for ( const Entry& entry : entries ) if ( strcmp( argv[1], entry.name ) == 0 ) sample = entry.create( &context );
	if ( !sample ) { fprintf( stderr, "unknown sample: %s\n", argv[1] ); return 2; }
	drivingControls = !strcmp( argv[1], "driving" );
	if ( drivingControls || !strcmp(argv[1], "geometricMover") ) context.camera.m_thirdPerson = true;
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
    characterDrawing = getenv("CHARACTER_DRAWING") ? fopen(getenv("CHARACTER_DRAWING"), "w") : nullptr;
    FILE* rigidReadouts = getenv("RIGID_READOUTS") ? fopen(getenv("RIGID_READOUTS"), "w") : nullptr;
    if (getenv("RIGID_READOUTS")) assert(rigidReadouts && !strcmp(argv[1], "rigidBody"));
    FILE* moverReadouts = getenv("MOVER_READOUTS") ? fopen(getenv("MOVER_READOUTS"), "w") : nullptr;
    if (getenv("MOVER_READOUTS")) assert(moverReadouts && !strcmp(argv[1], "geometricMover"));
    FILE* sensorReadouts = getenv("SENSOR_READOUTS") ? fopen(getenv("SENSOR_READOUTS"), "w") : nullptr;
    if (getenv("SENSOR_READOUTS")) assert(sensorReadouts && !strcmp(argv[1], "sensorHits"));
    bodyQueries = getenv("BODY_QUERIES") ? fopen(getenv("BODY_QUERIES"), "w") : nullptr;
    if (getenv("BODY_QUERIES")) assert(bodyQueries && !strcmp(argv[1], "cast"));
    FILE* readouts = getenv("PRECESSION_READOUTS") ? fopen(getenv("PRECESSION_READOUTS"), "w") : nullptr;
    if (getenv("PRECESSION_READOUTS")) assert(readouts && !strcmp(argv[1], "gyroscopicPrecession"));
    if (getenv("CONTACT_DRAWING")) {
        assert(!strcmp(argv[1], "persistentContact"));
        contactDrawing = fopen(getenv("CONTACT_DRAWING"), "w");
        assert(contactDrawing);
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
            } else if ( change.name == "dragX" ) context.camera.pickRay.origin.x = change.value;
            else if ( change.name == "dragY" ) context.camera.pickRay.origin.y = change.value;
            else if ( change.name == "dragZ" ) context.camera.pickRay.origin.z = change.value;
            else if ( change.name == "drag" ) {
                context.camera.pickRay.translation = {0,0,-20};
                if (change.value != 0) sample->MouseDown({}, 0, 0);
                else sample->MouseUp({}, 0);
            } else if ( change.name == "grabY" ) sample->m_mousePoint.y = change.value;
			else if ( change.name == "release" ) sample->Release();
		}
        sample->MouseMove({});
		currentStep = i;
		sample->DrawControls();
        contactMarks.clear();
        characterMarks.clear();
		sample->Step();
        if (contactDrawing) {
            fprintf(contactDrawing, "[");
            for (size_t j = 0; j < contactMarks.size(); ++j) {
                const auto& m = contactMarks[j];
                if (j) fprintf(contactDrawing, ",");
                fprintf(contactDrawing, "{\"point\":[%.9g,%.9g,%.9g],\"end\":[%.9g,%.9g,%.9g],\"label\":\"%s\"}", m.point.x,m.point.y,m.point.z,m.end.x,m.end.y,m.end.z,m.label.c_str());
            }
            fprintf(contactDrawing, "]\n");
        }
        if (rigidReadouts) {
            auto& c = static_cast<RigidBodyCharacter*>(sample)->m_character;
            auto p = b3Body_GetPosition(c.m_bodyId); auto vel = b3Body_GetLinearVelocity(c.m_bodyId);
            fprintf(rigidReadouts, "[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]\n", p.x,p.y,p.z,vel.x,vel.y,vel.z,c.m_onGround,c.m_sprint,c.m_jumpCooldown,c.m_groundNormal.x,c.m_groundNormal.y,c.m_groundNormal.z,c.m_groundVelocity.x,c.m_groundVelocity.y,c.m_groundVelocity.z,c.m_didStep,c.m_stepPosition.x,c.m_stepPosition.y,c.m_stepPosition.z,c.m_lastWishVelocity.x,c.m_lastWishVelocity.y,c.m_lastWishVelocity.z);
        }
        if (moverReadouts) {
            auto* scene = static_cast<GeometricMover*>(sample);
            auto& m = scene->m_mover;
            fprintf(moverReadouts, "[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%.9g,%.9g", m.m_transform.p.x,m.m_transform.p.y,m.m_transform.p.z,m.m_velocity.x,m.m_velocity.y,m.m_velocity.z,m.m_pogoVelocity,m.m_onGround,m.m_sprint,m.m_planeCount,m.m_totalIterations,scene->m_treeTimer,scene->m_respawnTimer);
            for (const auto* impact : {&scene->m_boxImpact, &scene->m_treeImpact})
                fprintf(moverReadouts, ",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d", impact->point.x,impact->point.y,impact->point.z,impact->normal.x,impact->normal.y,impact->normal.z,impact->speed,impact->valid);
            const auto& t = scene->m_impactTransform;
            fprintf(moverReadouts, ",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]\n", t.p.x,t.p.y,t.p.z,t.q.v.x,t.q.v.y,t.q.v.z,t.q.s);
        }
        if (sensorReadouts) {
            auto* sensor = static_cast<SensorHits*>(sample);
            fprintf(sensorReadouts, "[%d,%d,[", sensor->m_beginCount, sensor->m_endCount);
            for (int j = 0; j < sensor->m_transformCount; ++j) {
                b3Pos p = sensor->m_transforms[j].p;
                fprintf(sensorReadouts, "%s[%.9g,%.9g,%.9g]", j ? "," : "", p.x, p.y, p.z);
            }
            fprintf(sensorReadouts, "]]\n");
        }
        if (characterDrawing) {
            fprintf(characterDrawing, "[");
            for (size_t j = 0; j < characterMarks.size(); ++j) {
                fprintf(characterDrawing, "%s[", j ? "," : "");
                for (size_t k = 0; k < characterMarks[j].size(); ++k) fprintf(characterDrawing, "%s%.9g", k ? "," : "", characterMarks[j][k]);
                fprintf(characterDrawing, "]");
            }
            fprintf(characterDrawing, "]\n");
        }
        if (!strcmp(argv[1], "cast") || !strcmp(argv[1], "capsulePlane") || !strcmp(argv[1], "moverOverlap")) printf("%016llx\n", (unsigned long long)b3HashWorldState(b3GetWorldFromId(sample->m_worldId)));
        if (readouts) {
            auto* top = static_cast<GyroscopicPrecession*>(sample);
            if (top->m_elapsed > 0) fprintf(readouts, "%.9g %.9g\n", top->m_expectedAngle / top->m_elapsed, top->m_actualAngle / top->m_elapsed);
            else fprintf(readouts, "null null\n");
        }
        if ( recordQueriesAndBuilds ) sample->Render();
	}
    if (characterDrawing) fclose(characterDrawing);
    if (bodyQueries) fclose(bodyQueries);
    if (readouts) fclose(readouts);
    if (moverReadouts) fclose(moverReadouts);
    if (rigidReadouts) fclose(rigidReadouts);
    if (sensorReadouts) fclose(sensorReadouts);
    if (contactDrawing) fclose(contactDrawing);
	delete sample;
}
