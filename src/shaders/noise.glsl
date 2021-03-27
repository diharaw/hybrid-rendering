#ifndef NOISE_GLSL
#define NOISE_GLSL

//====
//note: normalized random, float=[0;1[
float nrand( vec2 n ) 
{
	return fract( sin(dot(n.xy, vec2(12.9898, 78.233)))* 43758.5453 );
}

vec2 nrand2( vec2 n ) 
{
	return fract( sin(dot(n.xy, vec2(12.9898, 78.233)))* vec2(43758.5453, 28001.8384) );
}

vec3 nrand3( vec2 n ) 
{
	return fract( sin(dot(n.xy, vec2(12.9898, 78.233)))* vec3(43758.5453, 28001.8384, 50849.4141 ) );
}

vec4 nrand4( vec2 n ) 
{
	return fract( sin(dot(n.xy, vec2(12.9898, 78.233)))* vec4(43758.5453, 28001.8384, 50849.4141, 12996.89) );
}

//====
//note: signed random, float=[-1;1[
float srand( vec2 n ) 
{
	return nrand( n ) * 2 - 1;
}

vec2 srand2( vec2 n ) 
{
	return nrand2( n ) * 2 - 1;
}

vec3 srand3( vec2 n ) 
{
	return nrand3( n ) * 2 - 1;
}

vec4 srand4( vec2 n ) 
{
	return nrand4( n ) * 2 - 1;
}

#endif