#include "HWAccelerator.hpp"
#include "VulkanFW/glfw/glfw.hpp"
#include "geom.h"
#include <iostream>

HWAccelerator::HWAccelerator(void)
{
    if (!glfwInit()) {
        std::cout << "Could not init GLFW" << std::endl;
        exit(1);
    }
    
    if (RT_TEST_WINDOW)
        setupWindow();
    else
        std::cout << "RT test: no window!" << std::endl;

    initVulkan();
    setupSwapchain();
    prepare();

    if (RT_TEST_WINDOW) {
        // TEST render loop
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Render frame
            if (prepared) {
                draw();
                enqueueTraceRays();
                finish();
                updateUniformBuffers();
            }
        }

        exit(1);
    }
    
    std::cout << "HWAccelerator init done" << std::endl;
}

void HWAccelerator::draw() {
    // Get next image in the swap chain (back/front buffer)
    prepareFrame();
    // Execute the compiled command buffer for the current swap chain image
    drawCurrentCommandBuffer();
    // Push the rendered frame to the surface
    submitFrame();
}

void HWAccelerator::prepareFrame() {
    // Acquire the next image from the swap chaing
    auto resultValue = swapChain.acquireNextImage(semaphores.acquireComplete);
    if (resultValue.result == vk::Result::eSuboptimalKHR) {
        glm::ivec2 newSize;
        glfwGetWindowSize(window, &newSize.x, &newSize.y);
        resultValue = swapChain.acquireNextImage(semaphores.acquireComplete);
    }
    currentBuffer = resultValue.value;
}

void HWAccelerator::submitFrame() {
    swapChain.queuePresent(semaphores.renderComplete);
}

void HWAccelerator::drawCurrentCommandBuffer() {
    vk::Fence fence = swapChain.getSubmitFence();
    {
        uint32_t fenceIndex = currentBuffer;
        context.dumpster.push_back([fenceIndex, this] { swapChain.clearSubmitFence(fenceIndex); });
    }

    // Command buffer(s) to be sumitted to the queue
    context.emptyDumpster(fence);
    {
        vk::SubmitInfo submitInfo;
        submitInfo.waitSemaphoreCount = (uint32_t)renderWaitSemaphores.size();
        submitInfo.pWaitSemaphores = renderWaitSemaphores.data();
        submitInfo.pWaitDstStageMask = renderWaitStages.data();

        submitInfo.signalSemaphoreCount = (uint32_t)renderSignalSemaphores.size();
        submitInfo.pSignalSemaphores = renderSignalSemaphores.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = commandBuffers.data() + currentBuffer;
        // Submit to queue
        context.queue.submit(submitInfo, fence);
    }

    context.recycle();
}

void HWAccelerator::debugPrintHit0()
{
    if (!hitBuffer.mapped)
        hitBuffer.map<Hit>(0, 1 * sizeof(Hit));
    
    Hit* hit0 = (Hit*)hitBuffer.mapped;
    std::cout << "Hit 0:\n"
        << "    i     " << hit0->i << std::endl
        << "    N     " << hit0->N << std::endl
        << "    P     " << hit0->P << std::endl
        << "    t     " << hit0->t << std::endl
        << "    uv    " << hit0->uvTex << std::endl
        << "    matID " << hit0->matId << std::endl;
}

void HWAccelerator::getRTDeviceInfo()
{
    auto props = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPropertiesNV>();
    auto devprops = props.get<vk::PhysicalDeviceProperties2>();
    auto rtprops = props.get<vk::PhysicalDeviceRayTracingPropertiesNV>();

    rtDevMaxRecursionDepth = rtprops.maxRecursionDepth;
    rtDevShaderGoupHandleSize = rtprops.shaderGroupHandleSize; // shaderHeaderSize

    std::cout << "Raytracing device (" << devprops.properties.deviceName << "):" << std::endl
        << "shaderGroupHandleSize: " << "\t" << rtprops.shaderGroupHandleSize << std::endl
        << "maxRecursionDepth: " << "\t" << rtprops.maxRecursionDepth << std::endl
        << "maxGeometryCount: " << "\t" << rtprops.maxGeometryCount << std::endl
        << std::endl;
}

void HWAccelerator::prepareTextureTarget(vks::Image & tex, uint32_t width, uint32_t height, vk::Format format)
{
    context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& setupCmdBuffer) {
        // Get device properties for the requested texture format
        vk::FormatProperties formatProperties;
        formatProperties = physicalDevice.getFormatProperties(format);
        // Check if requested image format supports image storage operations
        assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);

        // Prepare blit target texture
        tex.extent.width = width;
        tex.extent.height = height;

        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
        imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        imageCreateInfo.initialLayout = vk::ImageLayout::ePreinitialized;
        // vk::Image will be sampled in the fragment shader and used as storage target in the raytracing stage
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
        tex = context.createImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
        context.setImageLayout(setupCmdBuffer, tex.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::ePreinitialized, vk::ImageLayout::eGeneral);

        // Create sampler
        vk::SamplerCreateInfo sampler;
        sampler.magFilter = vk::Filter::eLinear;
        sampler.minFilter = vk::Filter::eLinear;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = vk::SamplerAddressMode::eRepeat;
        sampler.addressModeV = sampler.addressModeU;
        sampler.addressModeW = sampler.addressModeU;
        sampler.mipLodBias = 0.0f;
        sampler.maxAnisotropy = 0;
        sampler.compareOp = vk::CompareOp::eNever;
        sampler.minLod = 0.0f;
        sampler.maxLod = 0.0f;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        tex.sampler = device.createSampler(sampler);

        // Create image view
        vk::ImageViewCreateInfo view;
        view.viewType = vk::ImageViewType::e2D;
        view.format = format;
        view.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        view.image = tex.image;
        tex.view = device.createImageView(view);
    });
}

void HWAccelerator::updateDrawCommandBuffer(const vk::CommandBuffer & cmdBuffer)
{
    vk::ImageMemoryBarrier imageMemoryBarrier;
    imageMemoryBarrier.oldLayout = vk::ImageLayout::eGeneral;
    imageMemoryBarrier.newLayout = vk::ImageLayout::eGeneral;
    imageMemoryBarrier.image = textureRaytracingTarget.image;
    imageMemoryBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eInputAttachmentRead;
    cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, vk::DependencyFlags(), nullptr, nullptr,
        imageMemoryBarrier);
    cmdBuffer.setViewport(0, vks::util::viewport(size));
    cmdBuffer.setScissor(0, vks::util::rect2D(size));
    cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
    cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
    // Display ray traced image generated as a full screen quad
    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSetPostCompute, nullptr);
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.display);
    cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
}

// Rasterization
void HWAccelerator::setupDescriptorSetLayout() {
    std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
        // Binding 0 : Fragment shader image sampler
        { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
    };

    descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
    pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
}

// Rasterization
void HWAccelerator::setupDescriptorSet() {
    descriptorSetPostCompute = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
    updateDescriptorSets();
}

void HWAccelerator::updateDescriptorSets() {
    // vk::Image descriptor for the color map texture
    vk::DescriptorImageInfo texDescriptor{ textureRaytracingTarget.sampler, textureRaytracingTarget.view, vk::ImageLayout::eGeneral };

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
        // Binding 0 : Fragment shader texture sampler
        { descriptorSetPostCompute, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
    };

    device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void HWAccelerator::allocateCommandBuffers() {
    clearCommandBuffers();
    // Create one command buffer per image in the swap chain

    // Command buffers store a reference to the
    // frame buffer inside their render pass info
    // so for static usage without having to rebuild
    // them each frame, we use one per frame buffer
    commandBuffers = device.allocateCommandBuffers({ cmdPool, vk::CommandBufferLevel::ePrimary, swapChain.imageCount });
}

void HWAccelerator::clearCommandBuffers() {
    if (!commandBuffers.empty()) {
        context.trashCommandBuffers(cmdPool, commandBuffers);
        // FIXME find a better way to ensure that the draw and text buffers are no longer in use before
        // executing them within this command buffer.
        context.queue.waitIdle();
        context.device.waitIdle();
        context.recycle();
    }
}

void HWAccelerator::buildCommandBuffers() {
    // Destroy and recreate command buffers if already present
    allocateCommandBuffers();

    vk::CommandBufferBeginInfo cmdBufInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse };
    for (size_t i = 0; i < swapChain.imageCount; ++i) {
        const auto& cmdBuffer = commandBuffers[i];
        cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        cmdBuffer.begin(cmdBufInfo);
        //updateCommandBufferPreDraw(cmdBuffer); NOOP
        // Let child classes execute operations outside the renderpass, like buffer barriers or query pool operations
        renderPassBeginInfo.framebuffer = framebuffers[i];
        cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        updateDrawCommandBuffer(cmdBuffer);
        cmdBuffer.endRenderPass();
        //updateCommandBufferPostDraw(cmdBuffer); NOOP
        cmdBuffer.end();
    }
}

void HWAccelerator::updateRaytracingCommandBuffer()
{
    vk::CommandBufferBeginInfo cmdBufInfo;
    raytracingCmdBuffer.begin(cmdBufInfo);
    raytracingCmdBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, pipelines.raytracing);
    raytracingCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, raytracingPipelineLayout, 0, raytracingDescriptorSet, nullptr);

    auto missStride = rtDevShaderGoupHandleSize;
    auto hitStride = rtDevShaderGoupHandleSize;
    raytracingCmdBuffer.traceRaysNV(
        shaderBindingTable.buffer, // raygen SBT
        0,                         // raygen offset
        shaderBindingTable.buffer, 1 * rtDevShaderGoupHandleSize, missStride,  // miss SBT
        shaderBindingTable.buffer, 3 * rtDevShaderGoupHandleSize, hitStride,   // hit SBT
        nullptr, 0, 0, // callable SBT
        textureRaytracingTarget.extent.width,
        textureRaytracingTarget.extent.height,
        1, // depth
        loaderNV);

    raytracingCmdBuffer.end();
}

void HWAccelerator::createRaytracingCommandBuffer()
{
    raytracingCmdBuffer = device.allocateCommandBuffers({ cmdPool, vk::CommandBufferLevel::ePrimary, 1 })[0];
}

void HWAccelerator::prepareRasterizationPipeline() {
    vks::model::VertexLayout vertexLayoutQuad{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_COLOR,
    } };
    // Display pipeline
    vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout, renderPass };
    pipelineCreator.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    pipelineCreator.vertexInputState.appendVertexLayout(vertexLayoutQuad);
    pipelineCreator.loadShader("shaders/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
    pipelineCreator.loadShader("shaders/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);
    pipelines.display = pipelineCreator.create(context.pipelineCache);
}

void HWAccelerator::setupQueryPool()
{
    (void)rtPerfQueryPool;
}

void HWAccelerator::setupSharedBuffers()
{
    assert(uboRT.numTasks > 0);
    vk::DeviceSize hitBufferSize(uboRT.numTasks * sizeof(Hit));
    hitBuffer = context.createBuffer(vk::BufferUsageFlagBits::eRayTracingNV, vk::MemoryPropertyFlagBits::eHostVisible, hitBufferSize);
}

void HWAccelerator::buildAccelerationStructure()
{
    auto buildFlags = vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace;
    auto compactedSize = vk::DeviceSize(0);

    // Element counts
    uint32_t instanceCount = numInstancesNV; // top-level, number of instances of bottom-level structures
    uint32_t geometryCount = numMeshesNV;    // bottom-level

    auto topInfo = vk::AccelerationStructureInfoNV(vk::AccelerationStructureTypeNV::eTopLevel, buildFlags, instanceCount, 0, nullptr);
    auto bottomInfo = vk::AccelerationStructureInfoNV(vk::AccelerationStructureTypeNV::eBottomLevel, buildFlags, 0, geometryCount, geometries.data());

    topHandle = device.createAccelerationStructureNV(vk::AccelerationStructureCreateInfoNV(compactedSize, topInfo), nullptr, loaderNV);
    bottomHandle = device.createAccelerationStructureNV(vk::AccelerationStructureCreateInfoNV(compactedSize, bottomInfo), nullptr, loaderNV);

    // Get required sizes
    auto topLevelReq = vk::AccelerationStructureMemoryRequirementsInfoNV(vk::AccelerationStructureMemoryRequirementsTypeNV::eObject, topHandle);
    auto scratchReqTop = vk::AccelerationStructureMemoryRequirementsInfoNV(vk::AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch, topHandle);
    auto storageReqTop2 = device.getAccelerationStructureMemoryRequirementsNV(topLevelReq, loaderNV);
    auto scratchReqTop2 = device.getAccelerationStructureMemoryRequirementsNV(scratchReqTop, loaderNV);

    auto bottomLevelReq = vk::AccelerationStructureMemoryRequirementsInfoNV(vk::AccelerationStructureMemoryRequirementsTypeNV::eObject, bottomHandle);
    auto scratchReqBot = vk::AccelerationStructureMemoryRequirementsInfoNV(vk::AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch, bottomHandle);
    auto storageReqBot2 = device.getAccelerationStructureMemoryRequirementsNV(bottomLevelReq, loaderNV);
    auto scratchReqBot2 = device.getAccelerationStructureMemoryRequirementsNV(scratchReqBot, loaderNV);

    std::cout << "Top level storage requirement: " << storageReqTop2.memoryRequirements.size / 1024 << "KiB" << std::endl;
    std::cout << "Bottom level storage requirement: " << storageReqBot2.memoryRequirements.size / 1024 / 1024 << "MiB" << std::endl;

    // Scratch mem upper bound size
    vk::MemoryRequirements scratchReqs = scratchReqBot2.memoryRequirements;
    scratchReqs.size = std::max(scratchReqTop2.memoryRequirements.size, scratchReqBot2.memoryRequirements.size);

    scratchMem = context.createBuffer(vk::BufferUsageFlagBits::eRayTracingNV, vk::MemoryPropertyFlagBits::eDeviceLocal, scratchReqs.size);
    topLevelAccBuff = context.createBuffer(vk::BufferUsageFlagBits::eRayTracingNV, vk::MemoryPropertyFlagBits::eDeviceLocal, storageReqTop2.memoryRequirements.size);
    bottomLevelAccBuff = context.createBuffer(vk::BufferUsageFlagBits::eRayTracingNV, vk::MemoryPropertyFlagBits::eDeviceLocal, storageReqBot2.memoryRequirements.size);

    // Attach memory to acceleratio structures
    auto topMemoryInfo = vk::BindAccelerationStructureMemoryInfoNV(topHandle, topLevelAccBuff.memory, 0, 0, nullptr);
    device.bindAccelerationStructureMemoryNV(topMemoryInfo, loaderNV);
    auto bottomMemoryInfo = vk::BindAccelerationStructureMemoryInfoNV(bottomHandle, bottomLevelAccBuff.memory, 0, 0, nullptr);
    device.bindAccelerationStructureMemoryNV(bottomMemoryInfo, loaderNV);

    // Create instances
    for (int i = 0; i < numInstancesNV; i++) {
        InstanceNV instance{}; // zero init
        float identity[12] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f
        };
        memcpy(&instance.transform, identity, sizeof(identity));
        instance.instanceID = 0;
        instance.instanceMask = 0xff;
        instance.instanceContributionToHitGroupIndex = 0;
        instance.flags = (uint32_t)vk::GeometryInstanceFlagBitsNV::eTriangleCullDisable;
        if (vk::Result::eSuccess != device.getAccelerationStructureHandleNV(bottomHandle, sizeof(uint64_t), &instance.accelerationStructureHandle, loaderNV)) {
            throw std::exception();
        }
        instances.push_back(instance);
    }

    // Create instance buffer
    instancesNV = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eRayTracingNV, instances);

    // Build
    context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& commandBuffer) {
        auto bottomBuiltBarrier = vk::MemoryBarrier(vk::AccessFlagBits::eAccelerationStructureWriteNV, vk::AccessFlagBits::eAccelerationStructureReadNV); // src, dst

        // Build bottom level BVH
        auto bottomInfo = vk::AccelerationStructureInfoNV(vk::AccelerationStructureTypeNV::eBottomLevel, vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace, 0, numMeshesNV, geometries.data());
        commandBuffer.buildAccelerationStructureNV(
            bottomInfo,
            nullptr,           // instance data, must be null for BLAS
            0,                 // instance offset
            vk::Bool32(false), // update existing?
            bottomHandle,      // dst
            nullptr,           // src
            scratchMem.buffer, // scratch memory
            0,                 // scratch offset
            loaderNV
        );

        // Wait for bottom level build to finish
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildNV, vk::PipelineStageFlagBits::eAccelerationStructureBuildNV,
            vk::DependencyFlagBits::eDeviceGroup, bottomBuiltBarrier, nullptr, nullptr, loaderNV);

        // Build top level BVH
        auto topInfo = vk::AccelerationStructureInfoNV(vk::AccelerationStructureTypeNV::eTopLevel, vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace, numInstancesNV, 0, nullptr);
        commandBuffer.buildAccelerationStructureNV(
            topInfo,
            instancesNV.buffer, // instance data
            0,                  // instance offset
            vk::Bool32(false),  // update existing?
            topHandle,          // dst
            nullptr,            // src
            scratchMem.buffer,  // scratch memory
            0,                  // scratch offset
            loaderNV
        );
    });
}

void HWAccelerator::setupDescriptorPool()
{
    std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 2 }, // 1 for vertex shader, 1 for raygen shader
            // Graphics pipeline:
            { vk::DescriptorType::eCombinedImageSampler, 4 },
            // Raytracing pipeline:
            { vk::DescriptorType::eStorageImage, 1 },
            { vk::DescriptorType::eAccelerationStructureNV, 2 },
            { vk::DescriptorType::eStorageBuffer, 3 }, // hit shader: vertices, indices; rgen: hit buffer
    };

    descriptorPool = device.createDescriptorPool({ {}, /*maxSets*/3, (uint32_t)poolSizes.size(), poolSizes.data() });
}

void HWAccelerator::prepareRaytracing()
{
    // Descriptor set layout
    std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
        // Ray generation stage
        { 0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenNV },
        { 1, vk::DescriptorType::eAccelerationStructureNV, 1, vk::ShaderStageFlagBits::eRaygenNV },
        { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eRaygenNV }, // RT uniform buffer(camera params etc.)
        { 7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eRaygenNV }, // raygen hit buffer
        // Intersection stages
        { 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitNV }, // indices
        { 4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitNV }, // vertices
        { 5, vk::DescriptorType::eAccelerationStructureNV, 1, vk::ShaderStageFlagBits::eClosestHitNV },
        { 6, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eClosestHitNV },
    };


    // Pipeline layout
    raytracingDescriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
    raytracingPipelineLayout = device.createPipelineLayout({ {}, 1, &raytracingDescriptorSetLayout });

    // Pipeline
    std::vector<vk::PipelineShaderStageCreateInfo> rtStages;
    rtStages.push_back(vks::shaders::loadShader(device,
        "shaders/raygen.rgen.spv", vk::ShaderStageFlagBits::eRaygenNV, "main"));
    rtStages.push_back(vks::shaders::loadShader(device,
        "shaders/miss_primary.rmiss.spv", vk::ShaderStageFlagBits::eMissNV, "main"));
    rtStages.push_back(vks::shaders::loadShader(device,
        "shaders/miss_shadow.rmiss.spv", vk::ShaderStageFlagBits::eMissNV, "main"));
    rtStages.push_back(vks::shaders::loadShader(device,
        "shaders/diffuse.rchit.spv", vk::ShaderStageFlagBits::eClosestHitNV, "main"));
    rtStages.push_back(vks::shaders::loadShader(device,
        "shaders/shadow_blocker.rchit.spv", vk::ShaderStageFlagBits::eClosestHitNV, "main"));
    rtStages.push_back(vks::shaders::loadShader(device,
        "shaders/shadow_blocker.rahit.spv", vk::ShaderStageFlagBits::eAnyHitNV, "main"));

    assert(rtStages.size() == RT_STAGE_COUNT);

    /*
        Shader binding table:
        What   | Raygen | Miss primary ray | Miss shadow ray | Diffuse chit | Shadowray blocker chit | Shadowray blocker ahit |
        Group  | 0      | 1                | 2               | 3            | 4                      | 4                      |

        Hit shader index is calculated by:
        globalHitIndex = instanceShaderBindingTableRecordOffset (per instance) + hitProgramShaderBindingTableBaseIndex + geometryIndex × sbtRecordStride + sbtRecordOffset

        Miss shader index is given by:
        globalMissIndex = missIndex × sbtRecordStride + sbtRecordOffset
    */


    // eGeneral: contains a single raygen/miss/callable shader
    // eTrianglesHitGroup: must contain only chit and/or ahit (as such cannot hit non-triangle geometry)
    // eProceduralHitGroup: intersects custom geometry, must contain intersection shader, can contain chit and ahit
    const unsigned int EMPTY = VK_SHADER_UNUSED_NV;
    std::vector<vk::RayTracingShaderGroupCreateInfoNV> shaderGroups;
    shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoNV(vk::RayTracingShaderGroupTypeNV::eGeneral, 0, EMPTY, EMPTY, EMPTY)); // raygen
    shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoNV(vk::RayTracingShaderGroupTypeNV::eGeneral, 1, EMPTY, EMPTY, EMPTY)); // miss
    shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoNV(vk::RayTracingShaderGroupTypeNV::eGeneral, 2, EMPTY, EMPTY, EMPTY)); // miss
    shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoNV(vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup, EMPTY, 3, EMPTY, EMPTY)); // chit
    shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoNV(vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup, EMPTY, 4, 5, EMPTY)); // chit, ahit

    const uint32_t maxDepth = 2; // primary ray + shadow ray
    auto createInfo = vk::RayTracingPipelineCreateInfoNV({}, rtStages.size(), rtStages.data(), shaderGroups.size(), shaderGroups.data(), maxDepth, raytracingPipelineLayout);
    pipelines.raytracing = device.createRayTracingPipelineNV(context.pipelineCache, createInfo, nullptr, loaderNV);

    // Descriptor set
    auto sets = device.allocateDescriptorSets({ descriptorPool, 1, &raytracingDescriptorSetLayout });
    raytracingDescriptorSet = sets[0];
    updateRTDescriptorSets();
}

void HWAccelerator::updateRTDescriptorSets()
{
    auto accelInfo = vk::WriteDescriptorSetAccelerationStructureNV(1, &topHandle);
    std::vector<vk::DescriptorImageInfo> rtTexDescriptors{
        { nullptr, textureRaytracingTarget.view, vk::ImageLayout::eGeneral },
    };

    std::array<vk::WriteDescriptorSet, 8> rtWriteDescSets;
    rtWriteDescSets.at(0) = { raytracingDescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &rtTexDescriptors[0] };
    rtWriteDescSets.at(1) = { raytracingDescriptorSet, 1, 0, 1, vk::DescriptorType::eAccelerationStructureNV }; rtWriteDescSets.at(1).pNext = &accelInfo;
    rtWriteDescSets.at(2) = { raytracingDescriptorSet, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataRaytracing.descriptor };
    rtWriteDescSets.at(3) = { raytracingDescriptorSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &meshes.rtMesh.indices.descriptor };
    rtWriteDescSets.at(4) = { raytracingDescriptorSet, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &meshes.rtMesh.vertices.descriptor };
    rtWriteDescSets.at(5) = { raytracingDescriptorSet, 5, 0, 1, vk::DescriptorType::eAccelerationStructureNV }; rtWriteDescSets.at(5).pNext = &accelInfo;
    rtWriteDescSets.at(6) = { raytracingDescriptorSet, 6, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataRaytracing.descriptor };
    rtWriteDescSets.at(7) = { raytracingDescriptorSet, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &hitBuffer.descriptor };

    device.updateDescriptorSets(rtWriteDescSets, nullptr);
}

void HWAccelerator::setupShaderBindingTable()
{
    const uint32_t bindingTableSize = RT_GROUP_COUNT * rtDevShaderGoupHandleSize;
    const uint32_t firstGroup = 0;
    std::vector<char> opaqueHandles;
    opaqueHandles.resize(bindingTableSize);
    device.getRayTracingShaderGroupHandlesNV<char>(pipelines.raytracing, 0, RT_GROUP_COUNT, opaqueHandles, loaderNV);
    shaderBindingTable = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eRayTracingNV, opaqueHandles);
}

void HWAccelerator::updateUniformBuffers()
{
    uboRT.lightPos.x = 0.2f + sin(glm::radians(0.1f * glfwGetTime() * 360.0f)) * 1.0f;
    uboRT.lightPos.y = -0.8f;
    uboRT.lightPos.z = 0.0f;

    //uboRT.camPos = camera.position;
    glm::mat3 invR = glm::inverse(glm::mat3(/*camera.matrices.view*/));
    uboRT.invR = glm::mat4(invR);
    uboRT.aspectRatio = (float)size.width / (float)size.height;

    uniformDataRaytracing.copy(uboRT);
}

void HWAccelerator::getRaytracingQueue()
{
    uint32_t queueIndex = 0;

    std::vector<vk::QueueFamilyProperties> queueProps = physicalDevice.getQueueFamilyProperties();
    uint32_t queueCount = (uint32_t)queueProps.size();

    for (queueIndex = 0; queueIndex < queueCount; queueIndex++) {
        if (queueProps[queueIndex].queueFlags & vk::QueueFlagBits::eGraphics)
            break;
    }
    assert(queueIndex < queueCount);

    vk::DeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.queueFamilyIndex = queueIndex;
    queueCreateInfo.queueCount = 1;
    raytracingQueue = device.getQueue(queueIndex, 0);
}

void HWAccelerator::enqueueTraceRays()
{
    //raytracingCmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, rtPerfQueryPool, 1234, loaderNV);
    //static double lastPrinted = 0;

    //double t1 = glfwGetTime();

    vk::SubmitInfo raytracingSubmitInfo;
    raytracingSubmitInfo.commandBufferCount = 1;
    raytracingSubmitInfo.pCommandBuffers = &raytracingCmdBuffer;
    raytracingQueue.submit(raytracingSubmitInfo, nullptr);
    //raytracingQueue.waitIdle();

    //double t2 = glfwGetTime();

    // Print every 2 seconds
    //if (t2 - lastPrinted > 2.0) {
    //    double delta = t2 - t1;
    //    std::cout << "RT wallclock time: " << delta * 1000.0 << "ms" << std::endl;
    //    lastPrinted = t2;
    //}
    
}

void HWAccelerator::finish()
{
    raytracingQueue.waitIdle();
}

void HWAccelerator::setupWindow() {
    bool fullscreen = false;
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto monitor = glfwGetPrimaryMonitor();
    auto mode = glfwGetVideoMode(monitor);
    size.width = mode->width;
    size.height = mode->height;
    
    if (fullscreen) {
        window = glfwCreateWindow(size.width, size.height, "NVRay test window", monitor, nullptr);
    }
    else {
        size.width /= 2;
        size.height /= 2;
        window = glfwCreateWindow(size.width, size.height, "NVRay test window", nullptr, nullptr);
    }

    uboRT.numTasks = static_cast<unsigned int>(size.width * size.height);
    
    glfwSetWindowUserPointer(window, this);
    /*glfwSetKeyCallback(window, KeyboardHandler);
    glfwSetMouseButtonCallback(window, MouseHandler);
    glfwSetCursorPosCallback(window, MouseMoveHandler);
    glfwSetWindowCloseCallback(window, CloseHandler);
    glfwSetFramebufferSizeCallback(window, FramebufferSizeHandler);
    glfwSetScrollCallback(window, MouseScrollHandler);*/
    if (!window) {
        throw std::runtime_error("Could not create window");
    }
}

void HWAccelerator::setupSwapchain() {
    swapChain.setup(context.physicalDevice, context.device, context.queue, context.queueIndices.graphics);
    if (RT_TEST_WINDOW)
        swapChain.setSurface(surface);
}

void HWAccelerator::prepareUniformBuffers() {
    // Vertex shader uniform buffer block
    uniformDataRaytracing = context.createUniformBuffer(uboRT);
    updateUniformBuffers();
}

// TEST!
void HWAccelerator::loadAssets() {
    vks::model::VertexLayout vertexLayoutModel{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,
        vks::model::VERTEX_COMPONENT_COLOR,
        vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,
    } };

    // Setup geometry for raytraing
    vks::model::ModelCreateInfo modelCreateInfo;
    modelCreateInfo.scale = glm::vec3(0.1f, -0.1f, 0.1f);
    modelCreateInfo.uvscale = glm::vec2(1.0f);
    modelCreateInfo.center = glm::vec3(0.0f, 0.0f, 0.0f);

#ifdef _DEBUG 
    constexpr bool useDeer = true;
#else
    constexpr bool useDeer = false;
#endif

    if (useDeer) {
        modelCreateInfo.scale = glm::vec3(0.1f, -0.1f, -0.1f); // z: turn to face camera
        meshes.rtMesh.loadFromFile(context, "assets/lowpoly/deer.dae", vertexLayoutModel, modelCreateInfo);
        uboRT.camPos = glm::vec4(0.0f, 0.0f, -1.2f, 1.0f);
    }
    else {
        meshes.rtMesh.loadFromFile(context, "assets/sibenik/sibenik.dae", vertexLayoutModel, modelCreateInfo);
        uboRT.camPos = glm::vec4(0.5f, 0.0f, -0.5f, 1.0f);
    }
    
    const vk::IndexType meshIndexType = vk::IndexType::eUint32; // loadFromFile uses U32

    // Identity transformation for all meshes
    std::vector<float> Id3x4 = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    transform3x4 = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eUniformBuffer, Id3x4);

    const int numMeshes = 1;
    for (int i = 0; i < numMeshes; i++) {
        auto numVert = meshes.rtMesh.vertexCount;
        auto numInd = meshes.rtMesh.indexCount;
        auto strideVert = sizeof(VertexModel);
        auto tris = vk::GeometryTrianglesNV(meshes.rtMesh.vertices.buffer, 0, numVert, strideVert, vk::Format::eR32G32B32Sfloat, meshes.rtMesh.indices.buffer, 0, numInd, meshIndexType,
            transform3x4.buffer, 0);
        auto geomData = vk::GeometryDataNV(tris);                                          // union of tri and aabb, data read based on geometryTypeNV
        auto geomFlags = vk::GeometryFlagBitsNV::eOpaque;                                  // hits cannot be rejected (anyHit shader never run)
        auto geom = vk::GeometryNV(vk::GeometryTypeNV::eTriangles, geomData, geomFlags);   // type is triangles
        geometries.push_back(geom);
    }

    // Setup quad for drawing resulting image form raytracing pass
    struct VertexQuad {
        float pos[3];
        float uv[3];
    };
    const float dim = 1.0f;
    std::vector<VertexQuad> vertexBuffer = { { {  dim,  dim, 0.0f }, { 1.0f, 1.0f } },
                                             { { -dim,  dim, 0.0f }, { 0.0f, 1.0f } },
                                             { { -dim, -dim, 0.0f }, { 0.0f, 0.0f } },
                                             { {  dim, -dim, 0.0f }, { 1.0f, 0.0f } } };

    meshes.quad.vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
    std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
    meshes.quad.indexCount = (uint32_t)indexBuffer.size();
    meshes.quad.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
}

void HWAccelerator::prepare() {
    cmdPool = context.getCommandPool();
    setupSharedBuffers();

    if (RT_TEST_WINDOW) {
        swapChain.create(size, enableVsync);
        setupDepthStencil();
        setupRenderPass();
        setupRenderPassBeginInfo();
        setupFrameBuffer();
    }
    
    loadAssets();
    getRTDeviceInfo();
    loaderNV = vk::DispatchLoaderDynamic(context.instance, device);  // get NV function pointers at runtime
    getRaytracingQueue();
    buildAccelerationStructure();
    createRaytracingCommandBuffer();
    prepareUniformBuffers();
    prepareTextureTarget(textureRaytracingTarget, this->width, this->height, vk::Format::eR8G8B8A8Unorm);
    setupDescriptorSetLayout();
    
    if (RT_TEST_WINDOW)
        prepareRasterizationPipeline();
    
    setupDescriptorPool();
    setupDescriptorSet();
    prepareRaytracing();
    setupShaderBindingTable();
    
    if (RT_TEST_WINDOW)
        buildCommandBuffers();
    
    updateRaytracingCommandBuffer();
    prepared = true;
}

void HWAccelerator::setupDepthStencil()
{
    depthStencil.destroy();

    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    vk::ImageCreateInfo depthStencilCreateInfo;
    depthStencilCreateInfo.imageType = vk::ImageType::e2D;
    depthStencilCreateInfo.extent = vk::Extent3D{ size.width, size.height, 1 };
    depthStencilCreateInfo.format = depthFormat;
    depthStencilCreateInfo.mipLevels = 1;
    depthStencilCreateInfo.arrayLayers = 1;
    depthStencilCreateInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc;
    depthStencil = context.createImage(depthStencilCreateInfo);

    context.setImageLayout(depthStencil.image, aspect, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.viewType = vk::ImageViewType::e2D;
    depthStencilView.format = depthFormat;
    depthStencilView.subresourceRange.aspectMask = aspect;
    depthStencilView.subresourceRange.levelCount = 1;
    depthStencilView.subresourceRange.layerCount = 1;
    depthStencilView.image = depthStencil.image;
    depthStencil.view = device.createImageView(depthStencilView);
}

void HWAccelerator::setupRenderPass()
{
    if (renderPass) {
        device.destroyRenderPass(renderPass);
    }

    std::vector<vk::AttachmentDescription> attachments;
    attachments.resize(2);

    // Color attachment
    attachments[0].format = colorformat;
    attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
    attachments[0].initialLayout = vk::ImageLayout::eUndefined;
    attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

    // Depth attachment
    attachments[1].format = depthFormat;
    attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
    attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eClear;
    attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachments[1].initialLayout = vk::ImageLayout::eUndefined;
    attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    // Only one depth attachment, so put it first in the references
    vk::AttachmentReference depthReference;
    depthReference.attachment = 1;
    depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    std::vector<vk::AttachmentReference> colorAttachmentReferences;
    {
        vk::AttachmentReference colorReference;
        colorReference.attachment = 0;
        colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;
        colorAttachmentReferences.push_back(colorReference);
    }


    using vPSFB = vk::PipelineStageFlagBits;
    using vAFB = vk::AccessFlagBits;
    std::vector<vk::SubpassDependency> subpassDependencies{
        {
            0, VK_SUBPASS_EXTERNAL,
            vPSFB::eColorAttachmentOutput, vPSFB::eBottomOfPipe,
            vAFB::eColorAttachmentRead | vAFB::eColorAttachmentWrite, vAFB::eMemoryRead,
            vk::DependencyFlagBits::eByRegion
        },
        {
            VK_SUBPASS_EXTERNAL, 0,
            vPSFB::eBottomOfPipe, vPSFB::eColorAttachmentOutput,
            vAFB::eMemoryRead, vAFB::eColorAttachmentRead | vAFB::eColorAttachmentWrite,
            vk::DependencyFlagBits::eByRegion
        },
    };
    std::vector<vk::SubpassDescription> subpasses{
        {
            {}, vk::PipelineBindPoint::eGraphics,
            // Input attachment references
            0, nullptr,
            // Color / resolve attachment references
            (uint32_t)colorAttachmentReferences.size(), colorAttachmentReferences.data(), nullptr,
            // Depth stecil attachment reference,
            &depthReference,
            // Preserve attachments
            0, nullptr
        },
    };

    vk::RenderPassCreateInfo renderPassInfo;
    renderPassInfo.attachmentCount = (uint32_t)attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = (uint32_t)subpasses.size();
    renderPassInfo.pSubpasses = subpasses.data();
    renderPassInfo.dependencyCount = (uint32_t)subpassDependencies.size();
    renderPassInfo.pDependencies = subpassDependencies.data();
    renderPass = device.createRenderPass(renderPassInfo);
}

void HWAccelerator::setupRenderPassBeginInfo()
{
    clearValues.clear();
    clearValues.push_back(vks::util::clearColor(glm::vec4(0.1, 0.1, 0.1, 1.0)));
    clearValues.push_back(vk::ClearDepthStencilValue{ 1.0f, 0 });

    renderPassBeginInfo = vk::RenderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.extent = size;
    renderPassBeginInfo.clearValueCount = (uint32_t)clearValues.size();
    renderPassBeginInfo.pClearValues = clearValues.data();
}

void HWAccelerator::setupFrameBuffer()
{
    // Recreate the frame buffers
    if (!framebuffers.empty()) {
        for (uint32_t i = 0; i < framebuffers.size(); i++) {
            device.destroyFramebuffer(framebuffers[i]);
        }
        framebuffers.clear();
    }

    vk::ImageView attachments[2];

    // Depth/Stencil attachment is the same for all frame buffers
    attachments[1] = depthStencil.view;

    vk::FramebufferCreateInfo framebufferCreateInfo;
    framebufferCreateInfo.renderPass = renderPass;
    framebufferCreateInfo.attachmentCount = 2;
    framebufferCreateInfo.pAttachments = attachments;
    framebufferCreateInfo.width = size.width;
    framebufferCreateInfo.height = size.height;
    framebufferCreateInfo.layers = 1;

    // Create frame buffers for every swap chain image
    framebuffers = swapChain.createFramebuffers(framebufferCreateInfo);
}

void HWAccelerator::initVulkan()
{
    context.enableValidation = true;
    context.requireExtensions({ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME });
    context.requireDeviceExtensions({ VK_NV_RAY_TRACING_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME });

    context.setDeviceFeaturesPicker([this](const vk::PhysicalDevice& device, vk::PhysicalDeviceFeatures2& features) {
        if (deviceFeatures.textureCompressionBC) {
            enabledFeatures.textureCompressionBC = VK_TRUE;
        }
        else if (context.deviceFeatures.textureCompressionASTC_LDR) {
            enabledFeatures.textureCompressionASTC_LDR = VK_TRUE;
        }
        else if (context.deviceFeatures.textureCompressionETC2) {
            enabledFeatures.textureCompressionETC2 = VK_TRUE;
        }
        if (deviceFeatures.samplerAnisotropy) {
            enabledFeatures.samplerAnisotropy = VK_TRUE;
        }
        // getEnabledFeatures(); => returns nothing
    });

    if (RT_TEST_WINDOW) {
        // Window extensions not needed (not drawing UI w/ Vulkan)
        context.requireExtensions(glfw::Window::getRequiredInstanceExtensions());
        context.requireDeviceExtensions({ VK_KHR_SWAPCHAIN_EXTENSION_NAME });
    }
    
    context.createInstance(version);

    if (RT_TEST_WINDOW) {
        surface = glfw::Window::createWindowSurface(window, context.instance);
        context.createDevice(surface);

        // Find a suitable depth format
        depthFormat = context.getSupportedDepthFormat();
        // A semaphore used to synchronize image presentation
        // Ensures that the image is displayed before we start submitting new commands to the queu
        semaphores.acquireComplete = device.createSemaphore({});
        // A semaphore used to synchronize command submission
        // Ensures that the image is not presented until all commands have been sumbitted and executed
        semaphores.renderComplete = device.createSemaphore({});

        renderWaitSemaphores.push_back(semaphores.acquireComplete);
        renderWaitStages.push_back(vk::PipelineStageFlagBits::eBottomOfPipe);
        renderSignalSemaphores.push_back(semaphores.renderComplete);
    }
    else {
        context.createDevice();
    }
    

    
}
