// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#if defined(DEPTH_PROBE)
    #define PROBE_SIZE 16
    #define NUM_THREADS_X 8
    #define NUM_THREADS_Y 8
#else
    #define PROBE_SIZE 8
    #define NUM_THREADS_X 8
    #define NUM_THREADS_Y 4
#endif

#define PROBE_SIZE_WITH_BORDER (PROBE_SIZE + 2)

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout(local_size_x = NUM_THREADS_X, local_size_y = NUM_THREADS_Y, local_size_z = 1) in;

// ------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------
// ------------------------------------------------------------------

layout(set = 0, binding = 0, rgba16f) uniform image2D i_OutputIrradiance;
layout(set = 0, binding = 1, rg16f) uniform image2D i_OutputDepth;

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

#if defined(DEPTH_PROBE)
const ivec4 g_offsets[68] = ivec4[](
    ivec4(16, 1, 1, 0),
    ivec4(15, 1, 2, 0),
    ivec4(14, 1, 3, 0),
    ivec4(13, 1, 4, 0),
    ivec4(12, 1, 5, 0),
    ivec4(11, 1, 6, 0),
    ivec4(10, 1, 7, 0),
    ivec4(9, 1, 8, 0),
    ivec4(8, 1, 9, 0),
    ivec4(7, 1, 10, 0),
    ivec4(6, 1, 11, 0),
    ivec4(5, 1, 12, 0),
    ivec4(4, 1, 13, 0),
    ivec4(3, 1, 14, 0),
    ivec4(2, 1, 15, 0),
    ivec4(1, 1, 16, 0),
    ivec4(16, 16, 1, 17),
    ivec4(15, 16, 2, 17),
    ivec4(14, 16, 3, 17),
    ivec4(13, 16, 4, 17),
    ivec4(12, 16, 5, 17),
    ivec4(11, 16, 6, 17),
    ivec4(10, 16, 7, 17),
    ivec4(9, 16, 8, 17),
    ivec4(8, 16, 9, 17),
    ivec4(7, 16, 10, 17),
    ivec4(6, 16, 11, 17),
    ivec4(5, 16, 12, 17),
    ivec4(4, 16, 13, 17),
    ivec4(3, 16, 14, 17),
    ivec4(2, 16, 15, 17),
    ivec4(1, 16, 16, 17),
    ivec4(1, 16, 0, 1),
    ivec4(1, 15, 0, 2),
    ivec4(1, 14, 0, 3),
    ivec4(1, 13, 0, 4),
    ivec4(1, 12, 0, 5),
    ivec4(1, 11, 0, 6),
    ivec4(1, 10, 0, 7),
    ivec4(1, 9, 0, 8),
    ivec4(1, 8, 0, 9),
    ivec4(1, 7, 0, 10),
    ivec4(1, 6, 0, 11),
    ivec4(1, 5, 0, 12),
    ivec4(1, 4, 0, 13),
    ivec4(1, 3, 0, 14),
    ivec4(1, 2, 0, 15),
    ivec4(1, 1, 0, 16),
    ivec4(16, 16, 17, 1),
    ivec4(16, 15, 17, 2),
    ivec4(16, 14, 17, 3),
    ivec4(16, 13, 17, 4),
    ivec4(16, 12, 17, 5),
    ivec4(16, 11, 17, 6),
    ivec4(16, 10, 17, 7),
    ivec4(16, 9, 17, 8),
    ivec4(16, 8, 17, 9),
    ivec4(16, 7, 17, 10),
    ivec4(16, 6, 17, 11),
    ivec4(16, 5, 17, 12),
    ivec4(16, 4, 17, 13),
    ivec4(16, 3, 17, 14),
    ivec4(16, 2, 17, 15),
    ivec4(16, 1, 17, 16),
    ivec4(16, 16, 0, 0),
    ivec4(1, 16, 17, 0),
    ivec4(16, 1, 0, 17),
    ivec4(1, 1, 17, 17)
);
#else
const ivec4 g_offsets[36] = ivec4[](
    ivec4(8, 1, 1, 0),
    ivec4(7, 1, 2, 0),
    ivec4(6, 1, 3, 0),
    ivec4(5, 1, 4, 0),
    ivec4(4, 1, 5, 0),
    ivec4(3, 1, 6, 0),
    ivec4(2, 1, 7, 0),
    ivec4(1, 1, 8, 0),
    ivec4(8, 8, 1, 9),
    ivec4(7, 8, 2, 9),
    ivec4(6, 8, 3, 9),
    ivec4(5, 8, 4, 9),
    ivec4(4, 8, 5, 9),
    ivec4(3, 8, 6, 9),
    ivec4(2, 8, 7, 9),
    ivec4(1, 8, 8, 9),
    ivec4(1, 8, 0, 1),
    ivec4(1, 7, 0, 2),
    ivec4(1, 6, 0, 3),
    ivec4(1, 5, 0, 4),
    ivec4(1, 4, 0, 5),
    ivec4(1, 3, 0, 6),
    ivec4(1, 2, 0, 7),
    ivec4(1, 1, 0, 8),
    ivec4(8, 8, 9, 1),
    ivec4(8, 7, 9, 2),
    ivec4(8, 6, 9, 3),
    ivec4(8, 5, 9, 4),
    ivec4(8, 4, 9, 5),
    ivec4(8, 3, 9, 6),
    ivec4(8, 2, 9, 7),
    ivec4(8, 1, 9, 8),
    ivec4(8, 8, 0, 0),
    ivec4(1, 8, 9, 0),
    ivec4(8, 1, 0, 9),
    ivec4(1, 1, 9, 9)
);
#endif


// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

void copy_texel(ivec2 current_coord, uint index)
{
    ivec2 src_coord = current_coord + g_offsets[index].xy;
    ivec2 dst_coord = current_coord + g_offsets[index].zw;

#if defined(DEPTH_PROBE)    
    imageStore(i_OutputDepth, dst_coord, imageLoad(i_OutputDepth, src_coord));
#else   
    imageStore(i_OutputIrradiance, dst_coord, imageLoad(i_OutputIrradiance, src_coord));
#endif
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    const ivec2 current_coord = (ivec2(gl_WorkGroupID.xy) * ivec2(PROBE_SIZE_WITH_BORDER)) + ivec2(1);

    copy_texel(current_coord, gl_LocalInvocationIndex);

    if (gl_LocalInvocationIndex < 4)
        copy_texel(current_coord, (NUM_THREADS_X * NUM_THREADS_Y) + gl_LocalInvocationIndex);
}

// ------------------------------------------------------------------