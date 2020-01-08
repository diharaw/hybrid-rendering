#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.glsl"

layout (location = 0) rayPayloadInNV vec3 hit_value;

hitAttributeNV vec3 hit_attribs;

layout (set = 1, binding = 0) readonly buffer MaterialBuffer 
{
    uint id[];
} Material[];

layout (set = 1, binding = 1, std430) readonly buffer IndirectionBuffer 
{
    IndirectionInfo infos[];
} IndirectionArray;

layout (set = 1, binding = 2, std430) readonly buffer VertexBuffer 
{
    Vertex vertices[];
} VertexArray[];

layout (set = 1, binding = 3) readonly buffer IndexBuffer 
{
    uint indices[];
} IndexArray[];

layout(set = 2, binding = 0) uniform sampler2D s_Albedo[];
layout(set = 2, binding = 1) uniform sampler2D s_Normal[];
layout(set = 2, binding = 2) uniform sampler2D s_Roughness[];
layout(set = 2, binding = 3) uniform sampler2D s_Metallic[];

Vertex get_vertex(uint mesh_idx, uint vertex_idx)
{
    return VertexArray[nonuniformEXT(mesh_idx)].vertices[vertex_idx];
}

Vertex interpolated_attribs(uint mesh_idx)
{
    uvec3 idx = ivec3(IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID], 
                      IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID + 1],
                      IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID + 2]);

    Vertex v0 = get_vertex(mesh_idx, idx.x);
    Vertex v1 = get_vertex(mesh_idx, idx.y);
    Vertex v2 = get_vertex(mesh_idx, idx.z);

    const vec3 barycentrics = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);

    Vertex o;

    o.position.xyz = v0.position.xyz * barycentrics.x + v1.position.xyz * barycentrics.y + v2.position.xyz * barycentrics.z;
    o.tex_coord.xy = v0.tex_coord.xy * barycentrics.x + v1.tex_coord.xy * barycentrics.y + v2.tex_coord.xy * barycentrics.z;
    o.normal.xyz = normalize(v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z);
    o.tangent.xyz = normalize(v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z);
    o.bitangent.xyz = normalize(v0.bitangent.xyz * barycentrics.x + v1.bitangent.xyz * barycentrics.y + v2.bitangent.xyz * barycentrics.z);

    return o;
}

void main()
{
    const uint mat_idx = Material[0].id[3 * gl_PrimitiveID];

    Vertex v = interpolated_attribs(gl_InstanceCustomIndexNV);

    vec3 color = textureLod(s_Albedo[nonuniformEXT(mat_idx)], v.tex_coord.xy, 0.0).rgb;
    hit_value = color;
}
