#ifndef RANDOM_GLSL
#define RANDOM_GLSL

struct RNG
{
    uvec2 s;
};

// xoroshiro64* random number generator.
// http://prng.di.unimi.it/xoroshiro64star.c
uint rng_rotl(uint x, uint k)
{
    return (x << k) | (x >> (32 - k));
}

// Xoroshiro64* RNG
uint rng_next(inout RNG rng)
{
    uint result = rng.s.x * 0x9e3779bb;

    rng.s.y ^= rng.s.x;
    rng.s.x = rng_rotl(rng.s.x, 26) ^ rng.s.y ^ (rng.s.y << 9);
    rng.s.y = rng_rotl(rng.s.y, 13);

    return result;
}

// Thomas Wang 32-bit hash.
// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
uint rng_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

RNG rng_init(uvec2 id, uint frameIndex)
{
    uint s0 = (id.x << 16) | id.y;
    uint s1 = frameIndex;

    RNG rng;
    rng.s.x = rng_hash(s0);
    rng.s.y = rng_hash(s1);
    rng_next(rng);
    return rng;
}

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

#endif