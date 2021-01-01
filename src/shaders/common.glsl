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

#define LIGHT_DIRECTIONAL 0
#define LIGHT_POINT 1
#define LIGHT_SPOT 2

#define LIGHT_DIRECTION(val) val.xyz 
#define LIGHT_TYPE(val) 

struct RayPayload
{
    vec3 color;
};

struct VisibilityPayload
{
    bool visible;
};

struct Light
{
    vec4 data0;
    vec4 data1;
    vec4 data2;
    vec4 data3;
};

float light_type(in Light light)
{
    return light.data0.w;
}

vec3 light_direction(in Light light)
{
    return light.data0.xyz;
}

vec3 light_position(in Light light)
{
    return light.data2.xyz;
}

vec3 light_color(in Light light)
{
    return light.data3.xyz;
}

float light_intensity(in Light light)
{
    return light.data3.w;
}

float light_radius(in Light light)
{
    return light.data1.x;
}

float light_cos_theta_inner(in Light light)
{
    return light.data1.y;
}

float light_cos_theta_outer(in Light light)
{
    return light.data1.z;
}

#endif