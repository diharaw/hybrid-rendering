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

layout(set = 5, binding = 0) uniform samplerCube s_Cubemap;

// ------------------------------------------------------------------------
// PAYLOADS ---------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT IndirectDiffusePayload ray_payload;

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{	
	if (ray_payload.depth == 0)
		ray_payload.L = textureLod(s_Cubemap, gl_WorldRayDirectionEXT, 0.0f).rgb; 
	else
		ray_payload.L = vec3(0.0f);
}

// ------------------------------------------------------------------------