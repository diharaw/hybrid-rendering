#version 450

layout(set = 0, binding = 0) uniform sampler2D s_GBuffer1; // RGB: Albedo, A: Roughness
layout(set = 0, binding = 1) uniform sampler2D s_GBuffer2; // RGB: Normal, A: Metallic
layout(set = 0, binding = 2) uniform sampler2D s_GBuffer3; // RG: Motion Vector, BA: -
layout(set = 0, binding = 3) uniform sampler2D s_GBufferDepth;

layout(set = 1, binding = 0) uniform sampler2D s_Shadow;
layout(set = 2, binding = 0) uniform sampler2D s_Reflection;

layout(set = 3, binding = 0) uniform PerFrameUBO
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

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

#ifdef MOTION_VECTOR_DEBUG
const float PI = 3.1415927;

const int ARROW_V_STYLE    = 1;
const int ARROW_LINE_STYLE = 2;

// Choose your arrow head style
const int   ARROW_STYLE     = ARROW_LINE_STYLE;
const float ARROW_TILE_SIZE = 64.0;

// How sharp should the arrow head be? Used
const float ARROW_HEAD_ANGLE = 45.0 * PI / 180.0;

// Used for ARROW_LINE_STYLE
const float ARROW_HEAD_LENGTH     = ARROW_TILE_SIZE / 6.0;
const float ARROW_SHAFT_THICKNESS = 3.0;

// Computes the center pixel of the tile containing pixel pos
vec2 arrowTileCenterCoord(vec2 pos)
{
    return (floor(pos / ARROW_TILE_SIZE) + 0.5) * ARROW_TILE_SIZE;
}

// v = field sampled at tileCenterCoord(p), scaled by the length
// desired in pixels for arrows
// Returns 1.0 where there is an arrow pixel.
float arrow(vec2 p, vec2 v)
{
    // Make everything relative to the center, which may be fractional
    p -= arrowTileCenterCoord(p);

    float mag_v = length(v), mag_p = length(p);

    if (mag_v > 0.0)
    {
        // Non-zero velocity case
        vec2 dir_p = p / mag_p, dir_v = v / mag_v;

        // We can't draw arrows larger than the tile radius, so clamp magnitude.
        // Enforce a minimum length to help see direction
        mag_v = clamp(mag_v, 5.0, ARROW_TILE_SIZE / 2.0);

        // Arrow tip location
        v = dir_v * mag_v;

        // Define a 2D implicit surface so that the arrow is antialiased.
        // In each line, the left expression defines a shape and the right controls
        // how quickly it fades in or out.

        float dist;
        if (ARROW_STYLE == ARROW_LINE_STYLE)
        {
            // Signed distance from a line segment based on https://www.shadertoy.com/view/ls2GWG by
            // Matthias Reitinger, @mreitinger

            // Line arrow style
            dist = max(
                // Shaft
                ARROW_SHAFT_THICKNESS / 4.0 - max(abs(dot(p, vec2(dir_v.y, -dir_v.x))),                  // Width
                                                  abs(dot(p, dir_v)) - mag_v + ARROW_HEAD_LENGTH / 2.0), // Length

                // Arrow head
                min(0.0, dot(v - p, dir_v) - cos(ARROW_HEAD_ANGLE / 2.0) * length(v - p)) * 2.0 + // Front sides
                    min(0.0, dot(p, dir_v) + ARROW_HEAD_LENGTH - mag_v));                         // Back
        }
        else
        {
            // V arrow style
            dist = min(0.0, mag_v - mag_p) * 2.0 +                                                           // length
                min(0.0, dot(normalize(v - p), dir_v) - cos(ARROW_HEAD_ANGLE / 2.0)) * 2.0 * length(v - p) + // head sides
                min(0.0, dot(p, dir_v) + 1.0) +                                                              // head back
                min(0.0, cos(ARROW_HEAD_ANGLE / 2.0) - dot(normalize(v * 0.33 - p), dir_v)) * mag_v * 0.8;   // cutout
        }

        return clamp(1.0 + dist, 0.0, 1.0);
    }
    else
    {
        // Center of the pixel is always on the arrow
        return max(0.0, 1.2 - mag_p);
    }
}
#endif

void main()
{
    vec3  albedo     = texture(s_GBuffer1, inUV).rgb;
    vec3  normal     = texture(s_GBuffer2, inUV).rgb;
    vec3  reflection = texture(s_Reflection, inUV).rgb;
    float shadow     = texture(s_Shadow, inUV).r;

    //vec3 color = shadow * albedo * max(dot(normal, ubo.light_dir.xyz), 0.0) + albedo * 0.1 + reflection;
    vec3 color = shadow * vec3(max(dot(normal, ubo.light_dir.xyz), 0.0)) + vec3(0.1);

    // Reinhard tone mapping
    color = color / (1.0 + color);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

#ifdef MOTION_VECTOR_DEBUG
    vec2 motion_vector = normalize(texelFetch(s_GBuffer3, ivec2(arrowTileCenterCoord(gl_FragCoord.xy)), 0).rg);

    outFragColor = (1.0 - arrow(gl_FragCoord.xy, motion_vector * ARROW_TILE_SIZE * 0.4)) * vec4(color, 1.0);
#else
    outFragColor = vec4(color, 1.0);
#endif
}