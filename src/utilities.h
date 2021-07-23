#pragma once

#include <vk.h>

extern void                  pipeline_barrier(dw::vk::CommandBuffer::Ptr         cmd_buf,
                                              std::vector<VkMemoryBarrier>       memory_barriers,
                                              std::vector<VkImageMemoryBarrier>  image_memory_barriers,
                                              std::vector<VkBufferMemoryBarrier> buffer_memory_barriers,
                                              VkPipelineStageFlags               srcStageMask,
                                              VkPipelineStageFlags               dstStageMask);
extern VkImageMemoryBarrier  image_memory_barrier(dw::vk::Image::Ptr      image,
                                                  VkImageLayout           oldImageLayout,
                                                  VkImageLayout           newImageLayout,
                                                  VkImageSubresourceRange subresourceRange,
                                                  VkAccessFlags           srcAccessFlags,
                                                  VkAccessFlags           dstAccessFlags);
extern VkBufferMemoryBarrier buffer_memory_barrier(dw::vk::Buffer::Ptr buffer,
                                                   VkDeviceSize        offset,
                                                   VkDeviceSize        size,
                                                   VkAccessFlags       srcAccessFlags,
                                                   VkAccessFlags       dstAccessFlags);
extern VkMemoryBarrier       memory_barrier(VkAccessFlags srcAccessFlags, VkAccessFlags dstAccessFlags);