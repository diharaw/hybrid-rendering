#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.glsl"

layout(location = 0) rayPayloadInNV RayPayload ray_payload;

hitAttributeNV vec3 hit_attribs;

layout(set = 1, binding = 0) uniform PerFrameUBO
{
    mat4 view_inverse;
    mat4 proj_inverse;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 cam_pos;
    vec4 light_dir;
}
ubo;

layout (set = 3, binding = 0) readonly buffer MaterialBuffer 
{
    uint id[];
} Material[];

layout (set = 3, binding = 1, std430) readonly buffer VertexBuffer 
{
    Vertex vertices[];
} VertexArray[];

layout (set = 3, binding = 2) readonly buffer IndexBuffer 
{
    uint indices[];
} IndexArray[];

layout(set = 4, binding = 0) uniform sampler2D s_Albedo[];

layout(set = 5, binding = 0) uniform sampler2D s_Normal[];

layout(set = 6, binding = 0) uniform sampler2D s_Roughness[];

layout(set = 7, binding = 0) uniform sampler2D s_Metallic[];

Vertex get_vertex(uint mesh_idx, uint vertex_idx)
{
    return VertexArray[nonuniformEXT(mesh_idx)].vertices[vertex_idx];
}

Triangle fetch_triangle(uint mesh_idx)
{
    Triangle tri;

    uvec3 idx = uvec3(IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID], 
                      IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID + 1],
                      IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID + 2]);

    tri.v0 = get_vertex(mesh_idx, idx.x);
    tri.v1 = get_vertex(mesh_idx, idx.y);
    tri.v2 = get_vertex(mesh_idx, idx.z);

    tri.mat_idx = Material[nonuniformEXT(mesh_idx)].id[uint(tri.v0.position.w)];

    return tri;
}

Vertex interpolated_vertex(in Triangle tri)
{
    const vec3 barycentrics = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);

    Vertex o;

    o.position.xyz = tri.v0.position.xyz * barycentrics.x + tri.v1.position.xyz * barycentrics.y + tri.v2.position.xyz * barycentrics.z;
    o.tex_coord.xy = tri.v0.tex_coord.xy * barycentrics.x + tri.v1.tex_coord.xy * barycentrics.y + tri.v2.tex_coord.xy * barycentrics.z;
    o.normal.xyz = normalize(tri.v0.normal.xyz * barycentrics.x + tri.v1.normal.xyz * barycentrics.y + tri.v2.normal.xyz * barycentrics.z);
    o.tangent.xyz = normalize(tri.v0.tangent.xyz * barycentrics.x + tri.v1.tangent.xyz * barycentrics.y + tri.v2.tangent.xyz * barycentrics.z);
    o.bitangent.xyz = normalize(tri.v0.bitangent.xyz * barycentrics.x + tri.v1.bitangent.xyz * barycentrics.y + tri.v2.bitangent.xyz * barycentrics.z);

    return o;
}

vec3 get_normal_from_map(vec3 tangent, vec3 bitangent, vec3 normal, vec2 tex_coord, uint mat_idx)
{
    // Create TBN matrix.
    mat3 TBN = mat3(normalize(tangent), normalize(bitangent), normalize(normal));

    // Sample tangent space normal vector from normal map and remap it from [0, 1] to [-1, 1] range.
    vec3 n = normalize(textureLod(s_Normal[nonuniformEXT(mat_idx)], tex_coord, 0.0).rgb * 2.0 - 1.0);

    // Multiple vector by the TBN matrix to transform the normal from tangent space to world space.
    n = normalize(TBN * n);

    return n;
}

void main()
{
    const Triangle tri = fetch_triangle(gl_InstanceCustomIndexNV);
    const Vertex v = interpolated_vertex(tri);

    mat3 normal_mat = mat3(ubo.model);

    vec3 N = normal_mat * v.normal.xyz;
    vec3 T = normal_mat * v.tangent.xyz;
    vec3 B = normal_mat * v.bitangent.xyz;

    vec4 albedo = textureLod(s_Albedo[nonuniformEXT(tri.mat_idx)], v.tex_coord.xy, 0.0);
    vec3 normal = get_normal_from_map(T, B, N, v.tex_coord.xy, tri.mat_idx);

    vec3 color = albedo.rgb * max(dot(normal, ubo.light_dir.xyz), 0.0);
    float hit_distance = gl_HitTNV;

    if (albedo.a < 0.1)
    {
        color = vec3(0.0);
        hit_distance = 0;
    }

    ray_payload.color_dist = vec4(color, hit_distance);
}
