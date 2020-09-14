#ifndef COMMON_GLSL
#define COMMON_GLSL

#include "random.glsl"

#define RAY_GEN_SHADER_IDX 0
#define CLOSEST_HIT_SHADER_IDX 0
#define MISS_SHADER_IDX 0

#define PATH_TRACE_RAY_GEN_SHADER_IDX 0
#define PATH_TRACE_CLOSEST_HIT_SHADER_IDX 0
#define PATH_TRACE_MISS_SHADER_IDX 0

#define PRIMARY_RAY_PAYLOAD_LOC 0

#define M_PI 3.14159265359
#define EPSILON 0.0001f
#define INFINITY 100000.0f

#define MAX_RAY_BOUNCES 5

struct RayPayload
{
    vec4 color_dist;
};

struct PathTracePayload
{
    vec3 color;
    vec3 attenuation;
    float hit_distance;
    uint depth;
    RNG rng;
};

struct ShadowRayPayload
{
    float dist;
};

struct IndirectionInfo
{
    ivec2 idx;
};

struct Vertex
{
    vec4 position;
    vec4 tex_coord;
    vec4 normal;
    vec4 tangent;
    vec4 bitangent;
};

struct Triangle
{
    Vertex v0;
    Vertex v1;
    Vertex v2;
    uint mat_idx;
};

struct SurfaceProperties
{
    Vertex vertex;
    vec4 albedo;
    vec3 emissive;
    vec3 normal;
    vec3 F0;
    float metallic;
    float roughness;   
    float alpha;
    float alpha2; 
};

#endif