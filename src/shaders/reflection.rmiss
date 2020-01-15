#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) rayPayloadInNV RayPayload ray_payload;

void main()
{
    ray_payload.color_dist = vec4(0.0f);
}