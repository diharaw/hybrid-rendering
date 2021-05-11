#ifndef BND_SAMPLER_GLSL
#define BND_SAMPLER_GLSL

layout (set = 2, binding = 3, std430) readonly buffer BNDSobol 
{
    int data[];
} sobol_256spp_256d;

layout (set = 2, binding = 4, std430) readonly buffer BNDScramblingTile
{
    int data[];
} scrambling_tile;

layout (set = 2, binding = 5, std430) readonly buffer BNDRankingTile 
{
    int data[];
} ranking_tile;

float bnd_sampler(ivec2 coord, int sampleIndex, int sampleDimension)
{
	// wrap arguments
	coord.x = coord.x & 127;
	coord.y = coord.y & 127;
	sampleIndex = sampleIndex & 255;
	sampleDimension = sampleDimension & 255;

	// xor index based on optimized ranking
	int rankedSampleIndex = sampleIndex ^ ranking_tile.data[sampleDimension + (coord.x + coord.y*128)*8];

	// fetch value in sequence
	int value = sobol_256spp_256d.data[sampleDimension + rankedSampleIndex*256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ scrambling_tile.data[(sampleDimension%8) + (coord.x + coord.y*128)*8];

	// convert to float and return
	float v = (0.5f+value)/256.0f;
	return v;
}

#endif