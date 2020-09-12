#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) rayPayloadInNV PathTracePayload ray_payload;

void main()
{
    ray_payload.color = vec3(0.77f, 0.77f, 0.9f) * ray_payload.attenuation;
    ray_payload.hit_distance = 0.0f;
}