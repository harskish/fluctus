#pragma once

#include "VulkanFW/vks/vks.hpp"
#include "VulkanFW/vks/helpers.hpp"
#include "VulkanFW/vks/filesystem.hpp"
#include "VulkanFW/vks/shaders.hpp"
#include "VulkanFW/vks/model.hpp"
#include "VulkanFW/vks/pipelines.hpp"
#include "VulkanFW/vks/texture.hpp"

#ifdef WIN32
#include <handleapi.h>
#else
Linux / MacOS support still not implemented
#endif

#include <vulkan/vulkan.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>


// TEST: create window?
constexpr bool RT_TEST_WINDOW = true;


constexpr int RT_STAGE_COUNT = 6;
constexpr int RT_GROUP_COUNT = 5;

struct VertexModel {
    glm::vec4 pos;
    glm::vec4 normal;
    glm::vec4 color;
};


const uint32_t SHARED_TEXTURE_DIMENSION = 512;

class HWAccelerator
{
public:
    HWAccelerator(void);
    ~HWAccelerator(void) = default;

    void enqueueTraceRays();
    void finish();
    void transitionGL();
    void debugPrintHit0();
    void createGLObjects();

    void prepareFrame();

    unsigned int hitBufferSize() {
        return hitBuffer.allocSize;
    }

    struct GLHandles {
        GLuint semGlReady = 0;
        GLuint semGlComplete = 0;
        GLuint memShared = 0; // memory object EXT
        GLuint hitBuffer = 0; // buffer (GL_SHADER_STORAGE_BUFFER)
        GLuint memSharedTex = 0;
        GLuint color = 0; // TEST: shared texture

        GLuint vanillaColor = 0; // TEST: shared texture
        GLuint vanillaHitBuffer = 0;
    } glHandles;

    struct ShareHandles {
        HANDLE memory = INVALID_HANDLE_VALUE;
        HANDLE memory_tex = INVALID_HANDLE_VALUE;
        HANDLE glReady = INVALID_HANDLE_VALUE;
        HANDLE glComplete = INVALID_HANDLE_VALUE;
    } handles;

    

    //struct SharedResources {
    //    vks::Image texture;
    //    struct {
    //        vk::Semaphore glReady;
    //        vk::Semaphore glComplete;
    //    } semaphores;
    //    vk::CommandBuffer transitionCmdBuf;
    //    ShareHandles handles;
    //    vk::Device device;
    //
    //    void init(const vks::Context& context) {
    //        device = context.device;
    //        vk::DispatchLoaderDynamic dynamicLoader{ context.instance, device };
    //        {
    //            auto handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
    //            {
    //                vk::SemaphoreCreateInfo sci;
    //                vk::ExportSemaphoreCreateInfo esci;
    //                sci.pNext = &esci;
    //                esci.handleTypes = handleType;
    //                semaphores.glReady = device.createSemaphore(sci);
    //                semaphores.glComplete = device.createSemaphore(sci);
    //            }
    //            handles.glReady = device.getSemaphoreWin32HandleKHR({ semaphores.glReady, handleType }, dynamicLoader);
    //            handles.glComplete = device.getSemaphoreWin32HandleKHR({ semaphores.glComplete, handleType }, dynamicLoader);
    //        }
    //
    //        {
    //            vk::ImageCreateInfo imageCreateInfo;
    //            imageCreateInfo.imageType = vk::ImageType::e2D;
    //            imageCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
    //            imageCreateInfo.mipLevels = 1;
    //            imageCreateInfo.arrayLayers = 1;
    //            imageCreateInfo.extent.depth = 1;
    //            imageCreateInfo.extent.width = SHARED_TEXTURE_DIMENSION;
    //            imageCreateInfo.extent.height = SHARED_TEXTURE_DIMENSION;
    //            imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    //            texture.image = device.createImage(imageCreateInfo);
    //            texture.device = device;
    //            texture.format = imageCreateInfo.format;
    //            texture.extent = imageCreateInfo.extent;
    //        }
    //
    //        {
    //            vk::MemoryRequirements memReqs = device.getImageMemoryRequirements(texture.image);
    //            vk::MemoryAllocateInfo memAllocInfo;
    //            vk::ExportMemoryAllocateInfo exportAllocInfo{ vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 };
    //            memAllocInfo.pNext = &exportAllocInfo;
    //            memAllocInfo.allocationSize = texture.allocSize = memReqs.size;
    //            memAllocInfo.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    //            texture.memory = device.allocateMemory(memAllocInfo);
    //            device.bindImageMemory(texture.image, texture.memory, 0);
    //            handles.memory = device.getMemoryWin32HandleKHR({ texture.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 }, dynamicLoader);
    //        }
    //
    //        {
    //            // Create sampler
    //            vk::SamplerCreateInfo samplerCreateInfo;
    //            samplerCreateInfo.magFilter = vk::Filter::eLinear;
    //            samplerCreateInfo.minFilter = vk::Filter::eLinear;
    //            samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    //            // Max level-of-detail should match mip level count
    //            samplerCreateInfo.maxLod = (float)1;
    //            // Only enable anisotropic filtering if enabled on the devicec
    //            samplerCreateInfo.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
    //            samplerCreateInfo.anisotropyEnable = context.deviceFeatures.samplerAnisotropy;
    //            samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
    //            texture.sampler = device.createSampler(samplerCreateInfo);
    //        }
    //
    //        {
    //            // Create image view
    //            vk::ImageViewCreateInfo viewCreateInfo;
    //            viewCreateInfo.viewType = vk::ImageViewType::e2D;
    //            viewCreateInfo.image = texture.image;
    //            viewCreateInfo.format = texture.format;
    //            viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    //            texture.view = context.device.createImageView(viewCreateInfo);
    //        }
    //
    //        // Setup the command buffers used to transition the image between GL and VK
    //        transitionCmdBuf = context.createCommandBuffer();
    //        transitionCmdBuf.begin(vk::CommandBufferBeginInfo{});
    //        context.setImageLayout(transitionCmdBuf, texture.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
    //            vk::ImageLayout::eColorAttachmentOptimal);
    //        transitionCmdBuf.end();
    //    }
    //
    //    void destroy() {
    //        texture.destroy();
    //        device.destroy(semaphores.glComplete);
    //        device.destroy(semaphores.glReady);
    //    }
    //
    //    void transitionToGl(const vk::Queue& queue, const vk::Semaphore& waitSemaphore) const {
    //        vk::SubmitInfo submitInfo;
    //        vk::PipelineStageFlags stageFlags = vk::PipelineStageFlagBits::eBottomOfPipe;
    //        submitInfo.pWaitDstStageMask = &stageFlags;
    //        submitInfo.waitSemaphoreCount = 1;
    //        submitInfo.pWaitSemaphores = &waitSemaphore;
    //        submitInfo.signalSemaphoreCount = 1;
    //        submitInfo.pSignalSemaphores = &semaphores.glReady;
    //        submitInfo.commandBufferCount = 1;
    //        submitInfo.pCommandBuffers = &transitionCmdBuf;
    //        queue.submit({ submitInfo }, {});
    //    }
    //} shared;

private:
    void getRTDeviceInfo();
    void prepareTextureTarget(vks::Image& tex, uint32_t width, uint32_t height, vk::Format format);
    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer);
    void updateRaytracingCommandBuffer();
    void createRaytracingCommandBuffer();
    void buildAccelerationStructure();
    void setupDescriptorPool();
    void prepareRaytracing();
    void updateRTDescriptorSets();
    void setupShaderBindingTable();
    void loadAssets();
    void updateUniformBuffers();
    void getRaytracingQueue();

    void draw();
    // void prepareFrame(); => moved to public
    void submitFrame();
    void drawCurrentCommandBuffer();

private:
    struct InstanceNV {
        float transform[12];
        uint32_t instanceID : 24;
        uint32_t instanceMask : 8;
        uint32_t instanceContributionToHitGroupIndex : 24;
        uint32_t flags : 8;
        uint64_t accelerationStructureHandle;
    };

    // Order by size to avoid alignment mismatches between host and device
    struct UboCompute {
        glm::mat4 invR;
        glm::vec4 camPos = glm::vec4(0.5f, 0.0f, 0.0f, 1.0f);
        glm::vec4 lightPos;
        float aspectRatio;
        float fov = 90.0f;
        unsigned int numTasks = 0;
    } uboRT;

    struct {
        vks::model::Model quad; // for fragment shader
        vks::model::Model rtMesh; // for actual raytracing
    } meshes;

    struct {
        vk::Pipeline display;
        vk::Pipeline raytracing;
    } pipelines;

    vk::DispatchLoaderDynamic loaderNV;
    vks::Image textureRaytracingTarget;
    vks::Buffer uniformDataRaytracing;
    uint32_t rtDevMaxRecursionDepth = 0;
    uint32_t rtDevShaderGoupHandleSize = 0;
    uint32_t numMeshesNV = 1;
    uint32_t numInstancesNV = 1;
    std::vector<InstanceNV> instances;
    std::vector<vk::GeometryNV> geometries;
    vks::Buffer transform3x4;
    vks::Buffer scratchMem;
    vks::Buffer instancesNV;
    vks::Buffer shaderBindingTable;
    vks::Buffer topLevelAccBuff;
    vks::Buffer bottomLevelAccBuff;
    vk::AccelerationStructureNV topHandle;
    vk::AccelerationStructureNV bottomHandle;



    // Written to by Vulkan, accessed by OpenCL
    vks::Buffer hitBuffer;
    vks::Image texture;
    vk::CommandBuffer transitionCmdBuf;



    vk::Pipeline rtPipeline;
    vk::Queue raytracingQueue;
    vk::CommandBuffer raytracingCmdBuffer;
    vk::PipelineLayout raytracingPipelineLayout;
    vk::DescriptorSet raytracingDescriptorSet;
    vk::DescriptorSetLayout raytracingDescriptorSetLayout;

    vk::QueryPool rtPerfQueryPool;

    void setupWindow();
    void setupSwapchain();
    void initVulkan();
    void prepare();
    void setupDepthStencil();
    void setupRenderPass();
    void setupRenderPassBeginInfo();
    void setupFrameBuffer();
    void prepareUniformBuffers();
    void setupDescriptorSet();
    void setupDescriptorSetLayout();
    void updateDescriptorSets();
    void prepareRasterizationPipeline();
    void setupQueryPool();
    void setupSharedBuffers();
    void transitionToGl(const vk::Queue& queue, const vk::Semaphore& waitSemaphore) const;

    /* 
      ExampleBase
    */

    void allocateCommandBuffers();
    void buildCommandBuffers();
    void clearCommandBuffers();

    GLFWwindow* window{ nullptr };
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::ClearValue> clearValues;
    vk::RenderPassBeginInfo renderPassBeginInfo;
    vk::Viewport viewport() { return vks::util::viewport(size); }
    vk::Rect2D scissor() { return vks::util::rect2D(size); }
    vk::RenderPass renderPass;
    std::vector<vk::Framebuffer> framebuffers;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSetPostCompute;
    vk::DescriptorSetLayout descriptorSetLayout;

    // Color buffer format
    vk::Format colorformat{ vk::Format::eB8G8R8A8Unorm };

    // Depth buffer format...  selected during Vulkan initialization
    vk::Format depthFormat{ vk::Format::eUndefined };

    // Active frame buffer index
    uint32_t currentBuffer = 0;
    // Descriptor set pool
    vk::DescriptorPool descriptorPool;
    std::vector<vk::Semaphore> renderWaitSemaphores;
    std::vector<vk::PipelineStageFlags> renderWaitStages;
    std::vector<vk::Semaphore> renderSignalSemaphores;
    vks::Context context;
    const vk::PhysicalDevice& physicalDevice{ context.physicalDevice };
    const vk::Device& device{ context.device };
    const vk::Queue& queue{ context.queue };
    const vk::PhysicalDeviceFeatures& deviceFeatures{ context.deviceFeatures };
    vk::PhysicalDeviceFeatures& enabledFeatures{ context.enabledFeatures };
    //vkx::ui::UIOverlay ui{ context };
    vk::SurfaceKHR surface;
    // Wraps the swap chain to present images (framebuffers) to the windowing system
    vks::SwapChain swapChain;
    struct {
        vk::Semaphore acquireComplete; // swap chain image aquisition
        vk::Semaphore renderComplete;
        vk::Semaphore glReady;
        vk::Semaphore glComplete;
    } semaphores;
    // Command buffer pool
    
    vk::CommandPool cmdPool;
    bool prepared = false;
    uint32_t version = VK_MAKE_VERSION(1, 1, 0);
    vk::Extent2D size{ 1920, 1080 };
    uint32_t& width{ size.width };
    uint32_t& height{ size.height };

    vk::ClearColorValue defaultClearColor = vks::util::clearColor(glm::vec4({ 0.025f, 0.025f, 0.025f, 1.0f }));
    vk::ClearDepthStencilValue defaultClearDepth{ 1.0f, 0 };
    vks::Image depthStencil;

    bool enableVsync = false;
};