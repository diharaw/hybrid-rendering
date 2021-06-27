#ifndef GI_COMMON_GLSL
#define GI_COMMON_GLSL

#ifndef M_PI
#define M_PI 3.14159265359
#endif

// ------------------------------------------------------------------------

struct DDGIUniforms
{
    vec3  grid_start_position;
    vec3  grid_step;
    ivec3 probe_counts;
    float max_distance;
    float depth_sharpness;
    float hysteresis;
    float normal_bias;
    float energy_preservation;
    int   irradiance_probe_side_length;
    int   irradiance_texture_width;
    int   irradiance_texture_height;
    int   depth_probe_side_length;
    int   depth_texture_width;
    int   depth_texture_height;
    int   rays_per_probe;
};

// ------------------------------------------------------------------------

ivec3 base_grid_coord(in DDGIUniforms ddgi, vec3 X) 
{
    return clamp(ivec3((X - ddgi.grid_start_position) / ddgi.grid_step), ivec3(0, 0, 0), ivec3(ddgi.probe_counts) - ivec3(1, 1, 1));
}

// ------------------------------------------------------------------------

vec3 grid_coord_to_position(in DDGIUniforms ddgi, ivec3 c)
{
    return ddgi.grid_step * vec3(c) + ddgi.grid_start_position;
}

// ------------------------------------------------------------------------

int grid_coord_to_probe_index(in DDGIUniforms ddgi, in ivec3 probe_coords) 
{
    return int(probe_coords.x + probe_coords.y * ddgi.probe_counts.x + probe_coords.z * ddgi.probe_counts.x * ddgi.probe_counts.y);
}

// ------------------------------------------------------------------------

ivec3 probe_index_to_grid_coord(in DDGIUniforms ddgi, int index)
{
    ivec3 i_pos;

    // Slow, but works for any # of probes
    i_pos.x = index % ddgi.probe_counts.x;
    i_pos.y = (index % (ddgi.probe_counts.x * ddgi.probe_counts.y)) / ddgi.probe_counts.x;
    i_pos.z = index / (ddgi.probe_counts.x * ddgi.probe_counts.y);

    // Assumes probeCounts are powers of two.
    // Saves ~10ms compared to the divisions above
    // Precomputing the MSB actually slows this code down substantially
    //    i_pos.x = index & (ddgi.probe_counts.x - 1);
    //    i_pos.y = (index & ((ddgi.probe_counts.x * ddgi.probe_counts.y) - 1)) >> findMSB(ddgi.probe_counts.x);
    //    i_pos.z = index >> findMSB(ddgi.probe_counts.x * ddgi.probe_counts.y);

    return i_pos;
}

// ------------------------------------------------------------------------

vec3 probe_location(in DDGIUniforms ddgi, int index)
{
    // Compute grid coord from instance ID.
    ivec3 grid_coord = probe_index_to_grid_coord(ddgi, index);

    // Compute probe position from grid coord.
    return grid_coord_to_position(ddgi, grid_coord);
}

// ------------------------------------------------------------------

float square(float v)
{
    return v * v;
}

// ------------------------------------------------------------------

float pow3(float x) 
{ 
    return x * x * x; 
}

// ------------------------------------------------------------------------

float sign_not_zero(in float k)
{
    return (k >= 0.0) ? 1.0 : -1.0;
}

// ------------------------------------------------------------------

vec2 sign_not_zero(in vec2 v)
{
    return vec2(sign_not_zero(v.x), sign_not_zero(v.y));
}

// ------------------------------------------------------------------

vec2 oct_encode(in vec3 v) 
{
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    vec2 result = v.xy * (1.0 / l1norm);
    if (v.z < 0.0)
        result = (1.0 - abs(result.yx)) * sign_not_zero(result.xy);
    return result;
}

// ------------------------------------------------------------------

vec2 texture_coord_from_direction(vec3 dir, int probe_index, int full_texture_width, int full_texture_height, int probe_side_length) 
{
    vec2 normalized_oct_coord = oct_encode(normalize(dir));
    vec2 normalized_oct_coord_zero_one = (normalized_oct_coord + vec2(1.0f)) * 0.5f;

    // Length of a probe side, plus one pixel on each edge for the border
    float probe_with_border_side = float(probe_side_length) + 2.0f;

    vec2 oct_coord_normalized_to_texture_dimensions = (normalized_oct_coord_zero_one * float(probe_side_length)) / vec2(float(full_texture_width), float(full_texture_height));

    int probes_per_row = (full_texture_width - 2) / int(probe_with_border_side);

    // Add (2,2) back to texCoord within larger texture. Compensates for 1 pix 
    // border around texture and further 1 pix border around top left probe.
    vec2 probe_top_left_position = vec2(mod(probe_index, probes_per_row) * probe_with_border_side,
        (probe_index / probes_per_row) * probe_with_border_side) + vec2(2.0f, 2.0f);

    vec2 normalized_probe_top_left_position = vec2(probe_top_left_position) / vec2(float(full_texture_width), float(full_texture_height));

    return vec2(normalized_probe_top_left_position + oct_coord_normalized_to_texture_dimensions);
}

// ------------------------------------------------------------------------

vec3 sample_irradiance(in DDGIUniforms ddgi, vec3 P, vec3 N, sampler2D irradiance_texture, sampler2D depth_texture)
{
    ivec3 base_grid_coord = base_grid_coord(ddgi, P);
    ivec3 base_probe_pos = grid_coord_to_position(ddgi, base_grid_coord);
    
    vec3  sum_irradiance = vec3(0.0f);
    float sum_weight = 0.0f;

    // alpha is how far from the floor(currentVertex) position. on [0, 1] for each axis.
    vec3 alpha = clamp((P - base_probe_pos) / ddgi.grid_step, vec3(0.0f), vec3(1.0f));

    // Iterate over adjacent probe cage
    for (int i = 0; i < 8; ++i) 
    {
        // Compute the offset grid coord and clamp to the probe grid boundary
        // Offset = 0 or 1 along each axis
        ivec3  offset = ivec3(i, i >> 1, i >> 2) & ivec3(1);
        ivec3  probe_grid_coord = clamp(base_grid_coord + offset, ivec3(0), ddgi.probe_counts - ivec3(1));
        int p = grid_coord_to_probe_index(ddgi, probe_grid_coord);

        // Make cosine falloff in tangent plane with respect to the angle from the surface to the probe so that we never
        // test a probe that is *behind* the surface.
        // It doesn't have to be cosine, but that is efficient to compute and we must clip to the tangent plane.
        ivec3 probe_pos = grid_coord_to_position(ddgi, probe_grid_coord);

        // Bias the position at which visibility is computed; this
        // avoids performing a shadow test *at* a surface, which is a
        // dangerous location because that is exactly the line between
        // shadowed and unshadowed. If the normal bias is too small,
        // there will be light and dark leaks. If it is too large,
        // then samples can pass through thin occluders to the other
        // side (this can only happen if there are MULTIPLE occluders
        // near each other, a wall surface won't pass through itself.)
        vec3 probe_to_point = P - probe_pos + (N + 3.0 * w_o) * ddgi.normal_bias;
        vec3 dir = normalize(-probe_to_point);

        // Compute the trilinear weights based on the grid cell vertex to smoothly
        // transition between probes. Avoid ever going entirely to zero because that
        // will cause problems at the border probes. This isn't really a lerp. 
        // We're using 1-a when offset = 0 and a when offset = 1.
        vec3 trilinear = lerp(1.0 - alpha, alpha, offset);
        float weight = 1.0;

        // Clamp all of the multiplies. We can't let the weight go to zero because then it would be 
        // possible for *all* weights to be equally low and get normalized
        // up to 1/n. We want to distinguish between weights that are 
        // low because of different factors.

        // Smooth backface test
        {
            // Computed without the biasing applied to the "dir" variable. 
            // This test can cause reflection-map looking errors in the image
            // (stuff looks shiny) if the transition is poor.
            vec3 true_direction_to_probe = normalize(probe_pos - P);

            // The naive soft backface weight would ignore a probe when
            // it is behind the surface. That's good for walls. But for small details inside of a
            // room, the normals on the details might rule out all of the probes that have mutual
            // visibility to the point. So, we instead use a "wrap shading" test below inspired by
            // NPR work.
            // weight *= max(0.0001, dot(trueDirectionToProbe, wsN));

            // The small offset at the end reduces the "going to zero" impact
            // where this is really close to exactly opposite
            weight *= square(max(0.0001, (dot(true_direction_to_probe, N) + 1.0) * 0.5)) + 0.2;
        }

        // Moment visibility test
        {
            vec2 tex_coord = texture_coord_from_direction(-dir, p, ddgi.depth_texture_width, ddgi.depth_texture_height, ddgi.depth_probe_side_length);

            float dist_to_probe = length(probe_to_point);

            vec2 temp = texture(depth_texture, tex_coord, 0).rg;
            float mean = temp.x;
            float variance = abs(square(temp.x) - temp.y);

            // http://www.punkuser.net/vsm/vsm_paper.pdf; equation 5
            // Need the max in the denominator because biasing can cause a negative displacement
            float chebyshev_weight = variance / (variance + square(max(dist_to_probe - mean, 0.0)));
                
            // Increase contrast in the weight 
            chebyshev_weight = max(pow3(chebyshev_weight), 0.0);

            weight *= (dist_to_probe <= mean) ? 1.0 : chebyshev_weight;
        }

        // Avoid zero weight
        weight = max(0.000001, weight);
                 
        vec3 irradiance_dir = N;

        vec2 tex_coord = texture_coord_from_direction(normalize(irradiance_dir), p, ddgi.irradiance_texture_width, ddgi.irradiance_texture_height, ddgi.irradiance_probe_side_length);

        vec3 probe_irradiance = texture(irradiance_texture, tex_coord).rgb;

        // A tiny bit of light is really visible due to log perception, so
        // crush tiny weights but keep the curve continuous. This must be done
        // before the trilinear weights, because those should be preserved.
        const float crush_threshold = 0.2f;
        if (weight < crush_threshold)
            weight *= weight * weight * (1.0f / square(crush_threshold)); 

        // Trilinear weights
        weight *= trilinear.x * trilinear.y * trilinear.z;

        // Weight in a more-perceptual brightness space instead of radiance space.
        // This softens the transitions between probes with respect to translation.
        // It makes little difference most of the time, but when there are radical transitions
        // between probes this helps soften the ramp.
#       if LINEAR_BLENDING == 0
            probe_irradiance = sqrt(probe_irradiance);
#       endif
        
        sum_irradiance += weight * probe_irradiance;
        sum_weight += weight;
    }

    vec3 net_irradiance = sum_irradiance / sum_weight;

    // Go back to linear irradiance
#   if LINEAR_BLENDING == 0
        net_irradiance = square(net_irradiance);
#   endif
    net_irradiance *= ddgi.energy_preservation;

    return 0.5f * M_PI * net_irradiance;
}

// ------------------------------------------------------------------------

#endif