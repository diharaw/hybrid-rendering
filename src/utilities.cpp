#include "utilities.h"
#include <macros.h>

// -----------------------------------------------------------------------------------------------------------------------------------

void pipeline_barrier(dw::vk::CommandBuffer::Ptr         cmd_buf,
                      std::vector<VkMemoryBarrier>       memory_barriers,
                      std::vector<VkImageMemoryBarrier>  image_memory_barriers,
                      std::vector<VkBufferMemoryBarrier> buffer_memory_barriers,
                      VkPipelineStageFlags               srcStageMask,
                      VkPipelineStageFlags               dstStageMask)
{
    vkCmdPipelineBarrier(
        cmd_buf->handle(),
        srcStageMask,
        dstStageMask,
        0,
        memory_barriers.size(),
        memory_barriers.data(),
        buffer_memory_barriers.size(),
        buffer_memory_barriers.data(),
        image_memory_barriers.size(),
        image_memory_barriers.data());
}

// -----------------------------------------------------------------------------------------------------------------------------------

VkImageMemoryBarrier image_memory_barrier(dw::vk::Image::Ptr      image,
                                          VkImageLayout           oldImageLayout,
                                          VkImageLayout           newImageLayout,
                                          VkImageSubresourceRange subresourceRange,
                                          VkAccessFlags           srcAccessFlags,
                                          VkAccessFlags           dstAccessFlags)
{
    // Create an image barrier object
    VkImageMemoryBarrier memory_barrier;
    DW_ZERO_MEMORY(memory_barrier);

    memory_barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memory_barrier.oldLayout        = oldImageLayout;
    memory_barrier.newLayout        = newImageLayout;
    memory_barrier.image            = image->handle();
    memory_barrier.subresourceRange = subresourceRange;
    memory_barrier.srcAccessMask    = srcAccessFlags;
    memory_barrier.dstAccessMask    = dstAccessFlags;

    return memory_barrier;
}

// -----------------------------------------------------------------------------------------------------------------------------------

VkBufferMemoryBarrier buffer_memory_barrier(dw::vk::Buffer::Ptr buffer,
                                            VkDeviceSize        offset,
                                            VkDeviceSize        size,
                                            VkAccessFlags       srcAccessFlags,
                                            VkAccessFlags       dstAccessFlags)
{
    VkBufferMemoryBarrier memory_barrier;
    DW_ZERO_MEMORY(memory_barrier);

    memory_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = srcAccessFlags;
    memory_barrier.dstAccessMask = dstAccessFlags;
    memory_barrier.buffer        = buffer->handle();
    memory_barrier.offset        = offset;
    memory_barrier.size          = size;

    return memory_barrier;
}

// -----------------------------------------------------------------------------------------------------------------------------------

VkMemoryBarrier memory_barrier(VkAccessFlags srcAccessFlags, VkAccessFlags dstAccessFlags)
{
    // Create an image barrier object
    VkMemoryBarrier memory_barrier;
    DW_ZERO_MEMORY(memory_barrier);

    memory_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = srcAccessFlags;
    memory_barrier.dstAccessMask = dstAccessFlags;

    return memory_barrier;
}

// -----------------------------------------------------------------------------------------------------------------------------------