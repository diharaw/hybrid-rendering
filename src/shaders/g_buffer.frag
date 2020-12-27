#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene_descriptor_set.glsl"

// ------------------------------------------------------------------------
// INPUTS -----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) in vec3 FS_IN_FragPos;
layout(location = 1) in vec2 FS_IN_Texcoord;
layout(location = 2) in vec3 FS_IN_Normal;
layout(location = 3) in vec3 FS_IN_Tangent;
layout(location = 4) in vec3 FS_IN_Bitangent;
layout(location = 5) in vec4 FS_IN_CSPos;
layout(location = 6) in vec4 FS_IN_PrevCSPos;

// ------------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) out vec4 FS_OUT_GBuffer1; // RGB: Albedo, A: Roughness
layout(location = 1) out vec4 FS_OUT_GBuffer2; // RGB: Normal, A: Metallic
layout(location = 2) out vec4 FS_OUT_GBuffer3; // RG: Motion Vector, BA: -

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    mat4 model;
    uint material_idx;
} u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

vec2 motion_vector(vec4 prev_pos, vec4 current_pos)
{
    // Perspective division, covert clip space positions to NDC.
    vec2 current = (current_pos.xy / current_pos.w);
    vec2 prev    = (prev_pos.xy / prev_pos.w);

    // Remove jitter
    //current -= current_prev_jitter.xy;
    //prev -= current_prev_jitter.zw;

    // Remap to [0, 1] range
    current = current * 0.5 + 0.5;
    prev    = prev * 0.5 + 0.5;

    // Calculate velocity (prev -> current)
    return (current - prev);
}

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    const Material material = Materials.data[u_PushConstants.material_idx];

    vec4 albedo = fetch_albedo(material, FS_IN_Texcoord);

    if (albedo.a < 0.1)
        discard;

    // Albedo
    FS_OUT_GBuffer1.rgb = albedo.rgb;

    // Normal.
    FS_OUT_GBuffer2.rgb = fetch_normal(material, FS_IN_Tangent, FS_IN_Bitangent, FS_IN_Normal, FS_IN_Texcoord);

    // Roughness
    FS_OUT_GBuffer1.a = fetch_roughness(material, FS_IN_Texcoord);

    // Metallic
    FS_OUT_GBuffer2.a = fetch_metallic(material, FS_IN_Texcoord);

    // Motion Vector
    FS_OUT_GBuffer3.rg = motion_vector(FS_IN_PrevCSPos, FS_IN_CSPos);
}

// ------------------------------------------------------------------------