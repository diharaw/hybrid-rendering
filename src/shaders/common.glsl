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
#define RADIANCE_CLAMP_COLOR vec3(1.0f)

#define MAX_RAY_BOUNCES 5

struct RayPayload
{
    vec3 color;
};

struct ShadowRayPayload
{
    float dist;
};

#endif