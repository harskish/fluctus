#pragma once

#include "Kernel.hpp"
#include "tracer.hpp"
#include "clcontext.hpp"

inline CLContext* getCtxPtr(void* userPtr)
{
    Tracer* tracer = static_cast<Tracer*>(userPtr);
    CLContext *ctx = static_cast<CLContext*>(tracer->getClContext());
    return ctx;
}

class WFLogicKernel : public flt::Kernel
{
private:
    void setArgs() override {
        const CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks",          ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels",         ctx->deviceBuffers.pixelBuffer);
        err |= setArg("denoiserNormal", ctx->deviceBuffers.denoiserNormalBuffer);
        err |= setArg("denoiserAlbedo", ctx->deviceBuffers.denoiserAlbedoBuffer);
        err |= setArg("queueLens",      ctx->deviceBuffers.queueCounters);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("shadowQueue",    ctx->deviceBuffers.shadowQueue);
        err |= setArg("raygenQueue",    ctx->deviceBuffers.raygenQueue);
        err |= setArg("diffuseQueue",   ctx->deviceBuffers.diffuseMatQueue);
        err |= setArg("glossyQueue",    ctx->deviceBuffers.glossyMatQueue);
        err |= setArg("ggxReflQueue",   ctx->deviceBuffers.ggxReflMatQueue);
        err |= setArg("ggxRefrQueue",   ctx->deviceBuffers.ggxRefrMatQueue);
        err |= setArg("deltaQueue",     ctx->deviceBuffers.deltaMatQueue);
        err |= setArg("tris",           ctx->deviceBuffers.triangleBuffer);
        err |= setArg("nodes",          ctx->deviceBuffers.nodeBuffer);
        err |= setArg("indices",        ctx->deviceBuffers.indexBuffer);
        err |= setArg("envMap",         ctx->deviceBuffers.environmentMap);
        err |= setArg("probTable",      ctx->deviceBuffers.probTable);
        err |= setArg("aliasTable",     ctx->deviceBuffers.aliasTable);
        err |= setArg("pdfTable",       ctx->deviceBuffers.pdfTable);
        err |= setArg("materials",      ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData",        ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures",       ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("params",         ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks",       ctx->getNumTasks());
        err |= setArg("firstIteration", (cl_uint)false);
        verify(err, "Failed to set wf_logic arguments");
    }

    std::string getAdditionalBuildOptions() override {
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        const RenderParams& params = tracer->getParams();
        std::string opts;
        if (tracer->useDenoiser) opts.append(" -DUSE_OPTIX_DENOISER");
        if (params.useAreaLight) opts.append(" -DUSE_AREA_LIGHT");
        if (params.useEnvMap) opts.append(" -DUSE_ENV_MAP");
        if (params.sampleExpl) opts.append(" -DSAMPLE_EXPLICIT");
        if (params.sampleImpl) opts.append(" -DSAMPLE_IMPLICIT");
        return opts;
    }
};

class PickKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("tris", ctx->deviceBuffers.triangleBuffer);
        err |= setArg("nodes", ctx->deviceBuffers.nodeBuffer);
        err |= setArg("indices", ctx->deviceBuffers.indexBuffer);
        err |= setArg("pickResult", ctx->deviceBuffers.pickResult);
        verify(err, "Failed to set kernel_pick arguments!");
    }
};

class WFExtensionKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("tris", ctx->deviceBuffers.triangleBuffer);
        err |= setArg("nodes", ctx->deviceBuffers.nodeBuffer);
        err |= setArg("indices", ctx->deviceBuffers.indexBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_extension arguments!");
    }
};

class WFShadowKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("shadowQueue", ctx->deviceBuffers.shadowQueue);
        err |= setArg("tris", ctx->deviceBuffers.triangleBuffer);
        err |= setArg("nodes", ctx->deviceBuffers.nodeBuffer);
        err |= setArg("indices", ctx->deviceBuffers.indexBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_shadow arguments!");
    }
};

class WFRaygenKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("raygenQueue", ctx->deviceBuffers.raygenQueue);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("currPixelIdx", ctx->deviceBuffers.currentPixelIdx);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_raygen arguments!");
    }
};

class WFDiffuseKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("diffuseQueue", ctx->deviceBuffers.diffuseMatQueue);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_diffuse arguments!");
    }
};

class WFGlossyKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("glossyQueue", ctx->deviceBuffers.glossyMatQueue);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_glossy arguments!");
    }
};

class WFGGXReflKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("ggxReflQueue", ctx->deviceBuffers.ggxReflMatQueue);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_ggx_refl arguments!");
    }
};

class WFGGXRefrKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("ggxRefrQueue", ctx->deviceBuffers.ggxRefrMatQueue);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_ggx_refr arguments!");
    }
};

class WFDeltaKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("deltaQueue", ctx->deviceBuffers.deltaMatQueue);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_delta arguments!");
    }
};

class WFResetKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels", ctx->deviceBuffers.pixelBuffer);
        err |= setArg("denoiserAlbedo", ctx->deviceBuffers.denoiserAlbedoBuffer);
        err |= setArg("denoiserNormal", ctx->deviceBuffers.denoiserNormalBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("raygenQueue", ctx->deviceBuffers.raygenQueue);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set wf_reset arguments!");
    }
};

class MKResetKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels", ctx->deviceBuffers.pixelBuffer);
        err |= setArg("denoiserAlbedo", ctx->deviceBuffers.denoiserAlbedoBuffer);
        err |= setArg("denoiserNormal", ctx->deviceBuffers.denoiserNormalBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set mk_reset arguments!");
    }
};

class MKRaygenKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set mk_raygen arguments!");
    }
};

class MKNextVertexKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("denoiserNormal", ctx->deviceBuffers.denoiserNormalBuffer);
        err |= setArg("tris", ctx->deviceBuffers.triangleBuffer);
        err |= setArg("nodes", ctx->deviceBuffers.nodeBuffer);
        err |= setArg("indices", ctx->deviceBuffers.indexBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("stats", ctx->deviceBuffers.renderStats);
        err |= setArg("envMap", ctx->deviceBuffers.environmentMap);
        err |= setArg("pdfTable", ctx->deviceBuffers.pdfTable);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set mk_next_vertex arguments!");
    }

    std::string getAdditionalBuildOptions() override {
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        const RenderParams& params = tracer->getParams();
        std::string opts;
        if (tracer->useDenoiser) opts.append(" -DUSE_OPTIX_DENOISER");
        return opts;
    }
};

class MKSampleBSDFKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("denoiserAlbedo", ctx->deviceBuffers.denoiserAlbedoBuffer);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("envMap", ctx->deviceBuffers.environmentMap);
        err |= setArg("probTable", ctx->deviceBuffers.probTable);
        err |= setArg("aliasTable", ctx->deviceBuffers.aliasTable);
        err |= setArg("pdfTable", ctx->deviceBuffers.pdfTable);
        err |= setArg("tris", ctx->deviceBuffers.triangleBuffer);
        err |= setArg("nodes", ctx->deviceBuffers.nodeBuffer);
        err |= setArg("indices", ctx->deviceBuffers.indexBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("stats", ctx->deviceBuffers.renderStats);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set mk_sample_bsdf arguments!");
    }

    std::string getAdditionalBuildOptions() override {
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        const RenderParams& params = tracer->getParams();
        std::string opts;
        if (tracer->useDenoiser) opts.append(" -DUSE_OPTIX_DENOISER");
        return opts;
    }
};

class MKSplatKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels", ctx->deviceBuffers.pixelBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("stats", ctx->deviceBuffers.renderStats);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set mk_splat arguments!");
    }
};

class MKSplatPreviewKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels", ctx->deviceBuffers.pixelBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set mk_splat_preview arguments!");
    }
};

class MKPostprocessKernel : public flt::Kernel
{
private:
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("pixelsRaw", ctx->deviceBuffers.pixelBuffer); // raw pixels
        err |= setArg("denoiserAlbedo", ctx->deviceBuffers.denoiserAlbedoBuffer);
        err |= setArg("denoiserNormal", ctx->deviceBuffers.denoiserNormalBuffer);
        err |= setArg("pixelsPreview", ctx->deviceBuffers.previewBuffer); // tonemapped output
        err |= setArg("denoiserAlbedoGL", ctx->deviceBuffers.denoiserAlbedoBufferGL);
        err |= setArg("denoiserNormalGL", ctx->deviceBuffers.denoiserNormalBufferGL);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        verify(err, "Failed to set mk_postprocess arguments!");
    }

    std::string getAdditionalBuildOptions() override {
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        const RenderParams& params = tracer->getParams();
        std::string opts;
        if (tracer->useDenoiser) opts.append(" -DUSE_OPTIX_DENOISER");
        return opts;
    }
};