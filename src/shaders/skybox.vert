#version 450

layout(location = 0) in vec3 VS_IN_Position;
layout(location = 1) in vec3 VS_IN_Normal;
layout(location = 2) in vec2 VS_IN_TexCoord;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout(location = 0) out vec3 FS_IN_WorldPos;

// ------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------
// ------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    mat4 projection;
    mat4 view;
}
u_PushConstants;

// ------------------------------------------------------------------
// MAIN  ------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    FS_IN_WorldPos = VS_IN_Position;

    mat4 rotView = mat4(mat3(u_PushConstants.view));
    vec4 clipPos = u_PushConstants.projection * rotView * vec4(VS_IN_Position, 1.0);

    gl_Position = clipPos.xyww;
}