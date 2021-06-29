#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : require

#include "gi_common.glsl"

// ------------------------------------------------------------------------
// INPUTS -----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) in vec3 FS_IN_FragPos;
layout(location = 1) in vec3 FS_IN_Normal;
layout(location = 2) in flat int FS_IN_ProbeIdx;

// ------------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) out vec4 FS_OUT_Color;

// ------------------------------------------------------------------------
// DESCRIPTOR SET ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 1, binding = 0) uniform sampler2D s_Irradiance;
layout(set = 1, binding = 1) uniform sampler2D s_Depth;
layout(set = 1, binding = 2, scalar) uniform DDGIUBO
{
    DDGIUniforms ddgi; 
};

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    float scale;
}
u_PushConstants;

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    vec2 probe_coord = texture_coord_from_direction(normalize(FS_IN_Normal),
                                                    FS_IN_ProbeIdx,
                                                    ddgi.irradiance_texture_width,
                                                    ddgi.irradiance_texture_height,
                                                    ddgi.irradiance_probe_side_length);

    FS_OUT_Color = vec4(textureLod(s_Irradiance, probe_coord, 0.0f).rgb, 1.0f);
}

// ------------------------------------------------------------------------