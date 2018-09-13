#define NOMINMAX

#include "OptixDenoiser.hpp"
#include "window.hpp"

using namespace optix;

OptixDenoiser::OptixDenoiser(void)
{
    context = Context::create();
}

// Create RTBuffers using CUDA-GL sharing
// The buffers are now doubly shared (CUDA-GL and CL-GL)
void OptixDenoiser::bindBuffers(PTWindow * window)
{
    unsigned int width = window->getTexWidth();
    unsigned int height = window->getTexHeight();

    primal = context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, window->getPBO()); // must be IN/OUT due to CL_MEM_READ_WRITE?
    primal->setFormat(RT_FORMAT_FLOAT4);
    primal->setSize(width, height);
    context["input_buffer"]->set(primal);

    normals = context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, window->getNormalPBO());
    normals->setFormat(RT_FORMAT_FLOAT4);
    normals->setSize(width, height);
    context["input_normal_buffer"]->set(normals);

    albedos = context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, window->getAlbedoPBO());
    albedos->setFormat(RT_FORMAT_FLOAT4);
    albedos->setSize(width, height);
    context["input_albedo_buffer"]->set(albedos);

    setupCommandList(width, height);
}

// Called on framebuffer resize
void OptixDenoiser::resizeBuffers(PTWindow * window)
{
    unsigned int width = window->getTexWidth();
    unsigned int height = window->getTexHeight();

    auto resize = [width, height](optix::Buffer& buffer, GLuint pboId)
    {
        buffer->setSize(width, height);

        // Check if we have a GL interop display buffer
        if (pboId)
        {
            buffer->unregisterGLBuffer();
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            buffer->registerGLBuffer();
        }
    };

    resize(primal, window->getPBO());
    resize(normals, window->getNormalPBO());
    resize(albedos, window->getAlbedoPBO());
    setupCommandList(width, height);
}

// Perform denoising, write results to GL buffer
void OptixDenoiser::denoise(void)
{
    try
    {
        commandListWithDenoiser->execute();
    }
    catch (optix::Exception e)
    {
        std::cout << "OptixDenoiser error: " << e.what() << std::endl;
        waitExit();
    }
}

void OptixDenoiser::setupCommandList(unsigned int width, unsigned int height)
{
    try
    {
        if (!denoiserStage)
        {
            denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");
            denoiserStage->declareVariable("input_buffer")->set(primal);
            denoiserStage->declareVariable("output_buffer")->set(primal); // write over input
            denoiserStage->declareVariable("input_albedo_buffer");
            denoiserStage->declareVariable("input_normal_buffer");
            denoiserStage->declareVariable("blend")->setFloat(denoiseBlend);
            //denoiserStage->declareVariable("maxmem")->setFloat(1 /*GB*/ * 1e9f /*bytes per GB*/);

            if (useOptionalFeatures)
            {
                denoiserStage->queryVariable("input_albedo_buffer")->set(albedos);
                denoiserStage->queryVariable("input_normal_buffer")->set(normals);
            }
        }

        if (commandListWithDenoiser)
            commandListWithDenoiser->destroy();

        commandListWithDenoiser = context->createCommandList();
        commandListWithDenoiser->appendPostprocessingStage(denoiserStage, width, height);
        commandListWithDenoiser->finalize();

        context->validate();
    }
    catch (optix::Exception e)
    {
        std::cout << "OptixDenoiser error: " << e.what() << std::endl;
        waitExit();
    }
    
}
