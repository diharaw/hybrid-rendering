#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 4, binding = 0) uniform sampler2D s_IrradianceSH;
layout(set = 4, binding = 1) uniform samplerCube s_Prefiltered;
layout(set = 4, binding = 2) uniform sampler2D s_BRDF;

// ------------------------------------------------------------------------
// PAYLOADS ---------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT ReflectionPayload ray_payload;

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    const float MAX_REFLECTION_LOD = 4.0;
    ray_payload.color = textureLod(s_Prefiltered, gl_WorldRayDirectionEXT, ray_payload.roughness * MAX_REFLECTION_LOD).rgb;
    ray_payload.hit = false;
}

// ------------------------------------------------------------------------