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

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_POINT 1

struct RayPayload
{
    vec3 color;
};

struct ReflectionPayload
{
    vec3 color;
    vec3 hit_position;
    float ray_length;
    bool hit;
    RNG rng;
};

struct IndirectDiffusePayload
{
    vec3 L;
    vec3 T;
    uint depth;
    RNG rng;
};

struct Light
{
    vec4 data0;
    vec4 data1;
    ivec4 data2;
};

vec3 light_direction(in Light light)
{
    return light.data0.xyz;
}

vec3 light_color(in Light light)
{
    return light.data1.xyz;
}

float light_intensity(in Light light)
{
    return light.data0.w;
}

float light_radius(in Light light)
{
    return light.data1.w;
}

vec3 light_position(in Light light)
{
    return light.data0.xyz;
}

int light_type(in Light light)
{
    return light.data2.w;
}

#endif