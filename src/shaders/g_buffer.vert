#version 460

layout(location = 0) in vec3 VS_IN_Position;
layout(location = 1) in vec2 VS_IN_Texcoord;
layout(location = 2) in vec3 VS_IN_Normal;
layout(location = 3) in vec3 VS_IN_Tangent;
layout(location = 4) in vec3 VS_IN_Bitangent;

layout(location = 0) out vec3 FS_IN_FragPos;
layout(location = 1) out vec2 FS_IN_Texcoord;
layout(location = 2) out vec3 FS_IN_Normal;
layout(location = 3) out vec3 FS_IN_Tangent;
layout(location = 4) out vec3 FS_IN_Bitangent;
layout(location = 5) out vec4 FS_IN_CSPos;
layout(location = 6) out vec4 FS_IN_PrevCSPos;

layout(set = 0, binding = 0) uniform PerFrameUBO
{
    mat4 view_inverse;
    mat4 proj_inverse;
    mat4 view_proj_inverse;
    mat4 prev_view_proj;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 cam_pos;
    vec4 light_dir;
}
ubo;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    // Transform position into world space
    vec4 world_pos = ubo.model * vec4(VS_IN_Position, 1.0);

    // Transform world position into clip space
    gl_Position = ubo.projection * ubo.view * world_pos;

    // Pass world position into Fragment shader
    FS_IN_FragPos = world_pos.xyz;

    // Pass clip space positions for motion vectors
    FS_IN_CSPos     = gl_Position;
    FS_IN_PrevCSPos = ubo.prev_view_proj * world_pos;

    FS_IN_Texcoord = VS_IN_Texcoord;

    // Transform vertex normal into world space
    mat3 normal_mat = mat3(ubo.model);

    FS_IN_Normal    = normal_mat * VS_IN_Normal;
    FS_IN_Tangent   = normal_mat * VS_IN_Tangent;
    FS_IN_Bitangent = normal_mat * VS_IN_Bitangent;
}