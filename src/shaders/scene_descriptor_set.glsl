// ------------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------------
// ------------------------------------------------------------------------

struct Vertex
{
    vec4 position;
    vec4 tex_coord;
    vec4 normal;
    vec4 tangent;
    vec4 bitangent;
};

struct Triangle
{
    Vertex v0;
    Vertex v1;
    Vertex v2;
};

struct Material
{
    ivec4 texture_indices0; // x: albedo, y: normals, z: roughness, w: metallic
    ivec4 texture_indices1; // x: emissive, z: roughness_channel, w: metallic_channel
    vec4  albedo;
    vec4  emissive;
    vec4  roughness_metallic;
};

struct Instance
{
    mat4 model_matrix;
    uint mesh_idx;
};

struct HitInfo
{
    uint mat_idx;
    uint primitive_offset;
    uint primitive_id;
};

struct SurfaceProperties
{
    Vertex vertex;
    vec4 albedo;
    vec3 emissive;
    vec3 normal;
    vec3 F0;
    float metallic;
    float roughness;   
    float alpha;
    float alpha2; 
};

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout (set = 0, binding = 0, std430) readonly buffer MaterialBuffer 
{
    Material data[];
} Materials;

layout (set = 0, binding = 1, std430) readonly buffer InstanceBuffer 
{
    Instance data[];
} Instances;

#if defined(RAY_TRACING)
layout (set = 0, binding = 2) uniform accelerationStructureEXT u_TopLevelAS;
#endif

layout (set = 0, binding = 3, std430) readonly buffer VertexBuffer 
{
    Vertex data[];
} Vertices[1024];

layout (set = 0, binding = 4) readonly buffer IndexBuffer 
{
    uint data[];
} Indices[1024];

layout (set = 0, binding = 5) readonly buffer SubmeshInfoBuffer 
{
    uvec2 data[];
} SubmeshInfo[];

layout (set = 0, binding = 6) uniform sampler2D s_Textures[];

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

Vertex get_vertex(uint mesh_idx, uint vertex_idx)
{
    return Vertices[nonuniformEXT(mesh_idx)].data[vertex_idx];
}

// ------------------------------------------------------------------------

HitInfo fetch_hit_info(in Instance instance, in uint primitive_id, in uint geometry_index)
{
    uvec2 primitive_offset_mat_idx = SubmeshInfo[nonuniformEXT(instance.mesh_idx)].data[geometry_index];

    HitInfo hit_info;

    hit_info.mat_idx = primitive_offset_mat_idx.y;
    hit_info.primitive_offset = primitive_offset_mat_idx.x;
    hit_info.primitive_id = primitive_id;

    return hit_info;
}

// ------------------------------------------------------------------------

Triangle fetch_triangle(in Instance instance, in HitInfo hit_info)
{
    Triangle tri;

    uint primitive_id =  hit_info.primitive_id + hit_info.primitive_offset;

    uvec3 idx = uvec3(Indices[nonuniformEXT(instance.mesh_idx)].data[3 * primitive_id], 
                      Indices[nonuniformEXT(instance.mesh_idx)].data[3 * primitive_id + 1],
                      Indices[nonuniformEXT(instance.mesh_idx)].data[3 * primitive_id + 2]);

    tri.v0 = get_vertex(instance.mesh_idx, idx.x);
    tri.v1 = get_vertex(instance.mesh_idx, idx.y);
    tri.v2 = get_vertex(instance.mesh_idx, idx.z);

    return tri;
}

// ------------------------------------------------------------------------

Vertex interpolated_vertex(in Triangle tri, in vec3 barycentrics)
{;
    Vertex o;

    o.position = vec4(tri.v0.position.xyz * barycentrics.x + tri.v1.position.xyz * barycentrics.y + tri.v2.position.xyz * barycentrics.z, 1.0);
    o.tex_coord.xy = tri.v0.tex_coord.xy * barycentrics.x + tri.v1.tex_coord.xy * barycentrics.y + tri.v2.tex_coord.xy * barycentrics.z;
    o.normal.xyz = normalize(tri.v0.normal.xyz * barycentrics.x + tri.v1.normal.xyz * barycentrics.y + tri.v2.normal.xyz * barycentrics.z);
    o.tangent.xyz = normalize(tri.v0.tangent.xyz * barycentrics.x + tri.v1.tangent.xyz * barycentrics.y + tri.v2.tangent.xyz * barycentrics.z);
    o.bitangent.xyz = normalize(tri.v0.bitangent.xyz * barycentrics.x + tri.v1.bitangent.xyz * barycentrics.y + tri.v2.bitangent.xyz * barycentrics.z);

    return o;
}

// ------------------------------------------------------------------------

void transform_vertex(in Instance instance, inout Vertex v)
{
    mat4 model_mat = instance.model_matrix;
    mat3 normal_mat = mat3(instance.model_matrix);

    v.position = model_mat * v.position; 
    v.normal.xyz = normalize(normal_mat * v.normal.xyz);
    v.tangent.xyz = normalize(normal_mat * v.tangent.xyz);
    v.bitangent.xyz = normalize(normal_mat * v.bitangent.xyz);
}

// ------------------------------------------------------------------------

vec3 get_normal_from_map(vec3 tangent, vec3 bitangent, vec3 normal, vec2 tex_coord, uint normal_map_idx)
{
    // Create TBN matrix.
    mat3 TBN = mat3(normalize(tangent), normalize(bitangent), normalize(normal));

    // Sample tangent space normal vector from normal map and remap it from [0, 1] to [-1, 1] range.
    vec3 n = normalize(texture(s_Textures[nonuniformEXT(normal_map_idx)], tex_coord).rgb * 2.0 - 1.0);

    // Multiple vector by the TBN matrix to transform the normal from tangent space to world space.
    n = normalize(TBN * n);

    return n;
}

// ------------------------------------------------------------------------

vec4 fetch_albedo(in Material material, in vec2 texcoord)
{
    if (material.texture_indices0.x == -1)
        return material.albedo;
    else
        return texture(s_Textures[nonuniformEXT(material.texture_indices0.x)], texcoord);
}

// ------------------------------------------------------------------------

vec3 fetch_normal(in Material material, in vec3 tangent, in vec3 bitangent, in vec3 normal, in vec2 texcoord)
{
    if (material.texture_indices0.y == -1)
        return normal;
    else
        return get_normal_from_map(tangent, bitangent, normal, texcoord, material.texture_indices0.y);
}

// ------------------------------------------------------------------------

float fetch_roughness(in Material material, in vec2 texcoord)
{
    #define MIN_ROUGHNESS 0.1f

    if (material.texture_indices0.z == -1)
        return max(material.roughness_metallic.r, MIN_ROUGHNESS);
    else
        return max(texture(s_Textures[nonuniformEXT(material.texture_indices0.z)], texcoord)[material.texture_indices1.z], MIN_ROUGHNESS);
}

// ------------------------------------------------------------------------

float fetch_metallic(in Material material, in vec2 texcoord)
{
    if (material.texture_indices0.w == -1)
        return material.roughness_metallic.g;
    else
        return texture(s_Textures[nonuniformEXT(material.texture_indices0.w)], texcoord)[material.texture_indices1.w];
}

// ------------------------------------------------------------------------

vec3 fetch_emissive(in Material material, in vec2 texcoord)
{
    if (material.texture_indices1.x == -1)
        return material.emissive.rgb;
    else
        return texture(s_Textures[nonuniformEXT(material.texture_indices1.x)], texcoord).rgb;
}

// ------------------------------------------------------------------------