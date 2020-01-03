#version 450

layout(location = 0) in vec3 VS_IN_Position;
layout(location = 1) in vec2 VS_IN_Texcoord;
layout(location = 2) in vec3 VS_IN_Normal;
layout(location = 3) in vec3 VS_IN_Tangent;
layout(location = 4) in vec3 VS_IN_Bitangent;

layout (location = 0) out vec3 FS_IN_FragPos;
layout (location = 1) out vec2 FS_IN_Texcoord;
layout (location = 2) out vec3 FS_IN_Normal;

layout (set = 0, binding = 0) uniform PerFrameUBO 
{
	mat4 model;
	mat4 view;
	mat4 projection;
} ubo;

out gl_PerVertex
{
	vec4 gl_Position;
};

void main() 
{
    // Transform position into world space
	vec4 world_pos = ubo.model * vec4(VS_IN_Position, 1.0);

    // Pass world position into Fragment shader
    FS_IN_FragPos = world_pos.xyz;

    FS_IN_Texcoord = VS_IN_Texcoord;

    // Transform world position into clip space
	gl_Position = ubo.projection * ubo.view * world_pos;
	
    // Transform vertex normal into world space
    mat3 normal_mat = mat3(ubo.model);

	FS_IN_Normal = normal_mat * VS_IN_Normal;
}