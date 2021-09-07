#ifndef EDGE_STOPPING_GLSL
#define EDGE_STOPPING_GLSL

// ------------------------------------------------------------------------

#define DEPTH_FACTOR 0.5

// ------------------------------------------------------------------------

float normal_edge_stopping_weight(vec3 center_normal, vec3 sample_normal, float power)
{
    return pow(clamp(dot(center_normal, sample_normal), 0.0f, 1.0f), power);
}

// ------------------------------------------------------------------------

float depth_edge_stopping_weight(float center_lin_depth, float sample_lin_depth)
{
    float depth_diff = abs(center_lin_depth - sample_lin_depth);
    float d_factor   = depth_diff * DEPTH_FACTOR;
    return exp(-(d_factor * d_factor));
}

// ------------------------------------------------------------------

#endif