#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

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
layout(location = 1) out vec2 FS_IN_Texcoord;
layout(location = 2) out vec3 FS_IN_Normal;
layout(location = 3) out vec3 FS_IN_Tangent;
layout(location = 4) out vec3 FS_IN_Bitangent;
layout(location = 5) out vec4 FS_IN_CSPos;
layout(location = 6) out vec4 FS_IN_PrevCSPos;
layout(location = 7) out vec3 FS_IN_OSNormal;

out gl_PerVertex
{
    vec4 gl_Position;
};

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 1, binding = 0) uniform PerFrameUBO
{
    mat4  view_inverse;
    mat4  proj_inverse;
    mat4  view_proj_inverse;
    mat4  prev_view_proj;
    mat4  view_proj;
    vec4  cam_pos;
    Light light;
}
ubo;

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    mat4 model;
    mat4 prev_model;
    uint material_idx;
}
u_PushConstants;

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    // Transform position into world space
    vec4 world_pos      = u_PushConstants.model * vec4(VS_IN_Position, 1.0);
    vec4 prev_world_pos = u_PushConstants.prev_model * vec4(VS_IN_Position, 1.0);

    // Transform world position into clip space
    gl_Position = ubo.view_proj * world_pos;

    // Pass world position into Fragment shader
    FS_IN_FragPos = world_pos.xyz;

    // Pass clip space positions for motion vectors
    FS_IN_CSPos     = gl_Position;
    FS_IN_PrevCSPos = ubo.prev_view_proj * prev_world_pos;

    // Pass object space normal
    FS_IN_OSNormal = VS_IN_Normal;

    FS_IN_Texcoord = VS_IN_Texcoord;

    // Transform vertex normal into world space
    mat3 normal_mat = mat3(u_PushConstants.model);

    FS_IN_Normal    = normal_mat * VS_IN_Normal;
    FS_IN_Tangent   = normal_mat * VS_IN_Tangent;
    FS_IN_Bitangent = normal_mat * VS_IN_Bitangent;
}

// ------------------------------------------------------------------------