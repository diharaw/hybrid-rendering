#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL

#include "common.glsl"

float next_float(inout RNG rng)
{
    uint u = 0x3f800000 | (rng_next(rng) >> 9);
    return uintBitsToFloat(u) - 1.0;
}

uint next_uint(inout RNG rng, uint nmax)
{
    float f = next_float(rng);
    return uint(floor(f * nmax));
}

vec2 next_vec2(inout RNG rng)
{
    return vec2(next_float(rng), next_float(rng));
}

vec3 next_vec3(inout RNG rng)
{
    return vec3(next_float(rng), next_float(rng), next_float(rng));
}

mat3 make_rotation_matrix(vec3 z)
{
    const vec3 ref = abs(dot(z, vec3(0, 1, 0))) > 0.99f ? vec3(0, 0, 1) : vec3(0, 1, 0);

    const vec3 x = normalize(cross(ref, z));
    const vec3 y = cross(z, x);

    return mat3(x, y, z);
}

vec3 sample_cosine_lobe(in vec3 n, in vec2 r)
{
    vec2 rand_sample = max(vec2(0.00001f), r);

    const float phi = 2.0f * M_PI * rand_sample.y;

    const float cos_theta = sqrt(rand_sample.x);
    const float sin_theta = sqrt(1 - rand_sample.x);

    vec3 t = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    return normalize(make_rotation_matrix(n) * t);
}

#endif