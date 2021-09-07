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
#define LIGHT_TYPE_SPOT 2

#define MIRROR_REFLECTIONS_ROUGHNESS_THRESHOLD 0.05f
#define DDGI_REFLECTIONS_ROUGHNESS_THRESHOLD 0.75f

// ------------------------------------------------------------------------

struct RayPayload
{
    vec3 color;
};

// ------------------------------------------------------------------------

struct ReflectionPayload
{
    vec3 color;
    float ray_length;
};

// ------------------------------------------------------------------------

struct IndirectDiffusePayload
{
    vec3 L;
    vec3 T;
    uint depth;
    RNG rng;
};

// ------------------------------------------------------------------------

struct PathTracePayload
{
    vec3 L;
    vec3 T;
    uint depth;
    RNG rng;
};

// ------------------------------------------------------------------------

struct GIPayload
{
    vec3  L;
    vec3  T;
    float hit_distance;
    RNG   rng;
};

// ------------------------------------------------------------------------

struct Light
{
    vec4 data0;
    vec4 data1;
    vec4 data2;
    vec4 data3;
};

// ------------------------------------------------------------------------

vec3 light_direction(in Light light)
{
    return light.data0.xyz;
}

// ------------------------------------------------------------------------

vec3 light_color(in Light light)
{
    return light.data2.xyz;
}

// ------------------------------------------------------------------------

float light_intensity(in Light light)
{
    return light.data0.w;
}

// ------------------------------------------------------------------------

float light_radius(in Light light)
{
    return light.data1.w;
}

// ------------------------------------------------------------------------

vec3 light_position(in Light light)
{
    return light.data1.xyz;
}

// ------------------------------------------------------------------------

int light_type(in Light light)
{
    return int(light.data3.x);
}

// ------------------------------------------------------------------------

float light_cos_theta_outer(in Light light)
{
    return light.data3.y;
}

// ------------------------------------------------------------------------

float light_cos_theta_inner(in Light light)
{
    return light.data3.z;
}

// ------------------------------------------------------------------------

float luminance(vec3 rgb)
{
    return max(dot(rgb, vec3(0.299, 0.587, 0.114)), 0.0001);
}

// ------------------------------------------------------------------------

vec3 octohedral_to_direction(vec2 e)
{
    vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0)
        v.xy = (1.0 - abs(v.yx)) * (step(0.0, v.xy) * 2.0 - vec2(1.0));
    return normalize(v);
}

// ------------------------------------------------------------------

float gaussian_weight(float offset, float deviation)
{
    float weight = 1.0 / sqrt(2.0 * M_PI * deviation * deviation);
    weight *= exp(-(offset * offset) / (2.0 * deviation * deviation));
    return weight;
}

// ------------------------------------------------------------------------

vec3 world_position_from_depth(vec2 tex_coords, float ndc_depth, mat4 view_proj_inverse)
{
    // Take texture coordinate and remap to [-1.0, 1.0] range.
    vec2 screen_pos = tex_coords * 2.0 - 1.0;

    // // Create NDC position.
    vec4 ndc_pos = vec4(screen_pos, ndc_depth, 1.0);

    // Transform back into world position.
    vec4 world_pos = view_proj_inverse * ndc_pos;

    // Undo projection.
    world_pos = world_pos / world_pos.w;

    return world_pos.xyz;
}

// ------------------------------------------------------------------

float linear_eye_depth(float z, vec4 z_buffer_params)
{
    return 1.0 / (z_buffer_params.z * z + z_buffer_params.w);
}

// ------------------------------------------------------------------------

#endif