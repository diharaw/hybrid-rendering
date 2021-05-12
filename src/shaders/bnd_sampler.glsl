#ifndef BND_SAMPLER_GLSL
#define BND_SAMPLER_GLSL

float sample_blue_noise(ivec2 coord, int sample_index, int sample_dimension, sampler2D sobol_sequence_tex, sampler2D scrambling_ranking_tex)
{
	// wrap arguments
	coord.x = coord.x % 128;
	coord.y = coord.y % 128;
	sample_index = sample_index % 256;
	sample_dimension = sample_dimension % 4;

	// xor index based on optimized ranking
	int ranked_sample_index = sample_index ^ int(clamp(texelFetch(scrambling_ranking_tex, ivec2(coord.x, coord.y), 0).b * 256.0f, 0.0f, 255.0f));
	
    // fetch value in sequence
	int value = int(clamp(texelFetch(sobol_sequence_tex, ivec2(ranked_sample_index, 0), 0)[sample_dimension] * 256.0f, 0.0f, 255.0f));
	
    // If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ int(clamp(texelFetch(scrambling_ranking_tex, ivec2(coord.x, coord.y), 0)[sample_dimension % 2] * 256.0f, 0.0f, 255.0f));
	
    // convert to float and return
	float v = (0.5f + value) / 256.0f;
	return v;
}

#endif