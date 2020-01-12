#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout (location = 0) rayPayloadInNV ShadowRayPayload shadow_ray_payload;

void main()
{
    // In shadow
    shadow_ray_payload.dist = 0.0;
}
