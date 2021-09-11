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

float depth_edge_stopping_weight(float center_depth, float sample_depth, float phi)
{
    return exp(-abs(center_depth - sample_depth) / phi);
}

// ------------------------------------------------------------------

float luma_edge_stopping_weight(float center_luma, float sample_luma, float phi)
{
    return abs(center_luma - sample_luma) / phi;
}

// ------------------------------------------------------------------

float compute_edge_stopping_weight(
                      float center_depth,
                      float sample_depth,
                      float phi_z
                    #if defined(USE_EDGE_STOPPING_NORMAL_WEIGHT)
                    , vec3  center_normal,
                      vec3  sample_normal,
                      float phi_normal
                    #endif
                    #if defined(USE_EDGE_STOPPING_LUMA_WEIGHT)
                    , float center_luma,
                      float sample_luma,
                      float phi_luma
                    #endif
)
{
    const float wZ      = depth_edge_stopping_weight(center_depth, sample_depth, phi_z);
#if defined(USE_EDGE_STOPPING_NORMAL_WEIGHT)    
    const float wNormal = normal_edge_stopping_weight(center_normal, sample_normal, phi_normal);
#else
    const float wNormal = 1.0f;
#endif    
#if defined(USE_EDGE_STOPPING_LUMA_WEIGHT)
    const float wL      = luma_edge_stopping_weight(center_luma, sample_luma, phi_luma);
#else 
    const float wL = 1.0f;
#endif

    const float w = exp(0.0 - max(wL, 0.0) - max(wZ, 0.0)) * wNormal;

    return w;
}

// ------------------------------------------------------------------

#endif