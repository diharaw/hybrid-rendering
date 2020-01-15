#define RAY_GEN_SHADER_IDX 0
#define CLOSEST_HIT_SHADER_IDX 0
#define MISS_SHADER_IDX 0

#define PRIMARY_RAY_PAYLOAD_LOC 0

#define kPI 3.14159265359

struct RayPayload
{
    vec4 color_dist;
};

struct ShadowRayPayload
{
    float dist;
};

struct IndirectionInfo
{
    ivec2 idx;
};

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
    uint mat_idx;
};