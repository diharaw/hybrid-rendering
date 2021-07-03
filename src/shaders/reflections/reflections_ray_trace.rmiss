#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../common.glsl"

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 4, binding = 0) uniform samplerCube s_Cubemap;
layout(set = 4, binding = 1) uniform sampler2D s_IrradianceSH;
layout(set = 4, binding = 2) uniform samplerCube s_Prefiltered;
layout(set = 4, binding = 3) uniform sampler2D s_BRDF;

// ------------------------------------------------------------------------
// PAYLOADS ---------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT ReflectionPayload p_ReflectionPayload;

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    p_ReflectionPayload.color      = textureLod(s_Cubemap, gl_WorldRayDirectionEXT, 0.0f).rgb;
    p_ReflectionPayload.ray_length = -1.0f;
}

// ------------------------------------------------------------------------