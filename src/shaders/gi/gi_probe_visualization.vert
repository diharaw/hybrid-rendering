#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : require

#include "gi_common.glsl"
#include "../common.glsl"

// ------------------------------------------------------------------------
// INPUTS -----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) in vec3 VS_IN_Position;
layout(location = 1) in vec2 VS_IN_Texcoord;
layout(location = 2) in vec3 VS_IN_Normal;
layout(location = 3) in vec3 VS_IN_Tangent;
layout(location = 4) in vec3 VS_IN_Bitangent;

// ------------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) out vec3 FS_IN_FragPos;
layout(location = 1) out vec3 FS_IN_Normal;
layout(location = 2) out flat int FS_IN_ProbeIdx;

out gl_PerVertex
{
    vec4 gl_Position;
};

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 0, binding = 0) uniform PerFrameUBO
{
    mat4  view_inverse;
    mat4  proj_inverse;
    mat4  view_proj_inverse;
    mat4  prev_view_proj;
    mat4  view_proj;
    vec4  cam_pos;
    vec4  current_prev_jitter;
    Light light;
}
u_GlobalUBO;

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
    // Compute grid coord from instance ID.
    ivec3 grid_coord = probe_index_to_grid_coord(ddgi, gl_InstanceIndex);

    // Compute probe position from grid coord.
    vec3 probe_position = grid_coord_to_position(ddgi, grid_coord);

    // Scale and offset the vertex position.
    gl_Position = u_GlobalUBO.view_proj * vec4((VS_IN_Position * u_PushConstants.scale) + probe_position, 1.0f);

    // Pass normal and probe index into the fragment shader.
    FS_IN_Normal   = VS_IN_Normal;
    FS_IN_ProbeIdx = gl_InstanceIndex;
}

// ------------------------------------------------------------------------