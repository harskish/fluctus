#pragma once

#include <clt.hpp>
#include "tracer.hpp"
#include "clcontext.hpp"

inline CLContext* getCtxPtr(void* userPtr)
{
    Tracer* tracer = static_cast<Tracer*>(userPtr);
    CLContext *ctx = static_cast<CLContext*>(tracer->getClContext());
    return ctx;
}

class WFLogicKernel : public clt::Kernel
{
public:
    WFLogicKernel(void) : Kernel("src/wf_logic.cl", "logic") {}
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
        clt::check(err, "Failed to set wf_logic arguments");
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
        if (!params.wfSeparateQueues) opts.append(" -DWF_SINGLE_MAT_QUEUE");
        return opts;
    }
};

class PickKernel : public clt::Kernel
{
public:
    PickKernel(void) : Kernel("src/kernel_pick.cl", "pick") {}
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("tris", ctx->deviceBuffers.triangleBuffer);
        err |= setArg("nodes", ctx->deviceBuffers.nodeBuffer);
        err |= setArg("indices", ctx->deviceBuffers.indexBuffer);
        err |= setArg("pickResult", ctx->deviceBuffers.pickResult);
        clt::check(err, "Failed to set kernel_pick arguments!");
    }
};

class WFExtensionKernel : public clt::Kernel
{
public:
    WFExtensionKernel(void) : Kernel("src/wf_extrays.cl", "traceExtension") {}
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
        clt::check(err, "Failed to set wf_extension arguments!");
    }
};

class WFShadowKernel : public clt::Kernel
{
public:
    WFShadowKernel(void) : Kernel("src/wf_shadowrays.cl", "traceShadow") {}
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
        clt::check(err, "Failed to set wf_shadow arguments!");
    }
};

class WFRaygenKernel : public clt::Kernel
{
public:
    WFRaygenKernel(void) : Kernel("src/wf_raygen.cl", "genRays") {}
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
        clt::check(err, "Failed to set wf_raygen arguments!");
    }
};

class WFDiffuseKernel : public clt::Kernel
{
public:
    WFDiffuseKernel(void) : Kernel("src/wf_mat_diffuse.cl", "wavefrontDiffuse") {}
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
        clt::check(err, "Failed to set wf_diffuse arguments!");
    }
};

class WFGlossyKernel : public clt::Kernel
{
public:
    WFGlossyKernel(void) : Kernel("src/wf_mat_glossy.cl", "wavefrontGlossy") {}
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
        clt::check(err, "Failed to set wf_glossy arguments!");
    }
};

class WFGGXReflKernel : public clt::Kernel
{
public:
    WFGGXReflKernel(void) : Kernel("src/wf_mat_ggx_reflection.cl", "wavefrontGGXReflection") {}
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
        clt::check(err, "Failed to set wf_ggx_refl arguments!");
    }
};

class WFGGXRefrKernel : public clt::Kernel
{
public:
    WFGGXRefrKernel(void) : Kernel("src/wf_mat_ggx_refraction.cl", "wavefrontGGXRefraction") {}
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
        clt::check(err, "Failed to set wf_ggx_refr arguments!");
    }
};

class WFDeltaKernel : public clt::Kernel
{
public:
    WFDeltaKernel(void) : Kernel("src/wf_mat_delta.cl", "wavefrontDelta") {}
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
        clt::check(err, "Failed to set wf_delta arguments!");
    }
};

class WFAllMaterialsKernel : public clt::Kernel
{
public:
    WFAllMaterialsKernel(void) : Kernel("src/wf_mat_all.cl", "wavefrontAllMaterials") {}
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("queueLens", ctx->deviceBuffers.queueCounters);
        err |= setArg("materialQueue", ctx->deviceBuffers.diffuseMatQueue);
        err |= setArg("extensionQueue", ctx->deviceBuffers.extensionQueue);
        err |= setArg("materials", ctx->deviceBuffers.materialBuffer);
        err |= setArg("texData", ctx->deviceBuffers.texDataBuffer);
        err |= setArg("textures", ctx->deviceBuffers.texDescriptorBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        clt::check(err, "Failed to set wf_all_mats arguments!");
    }

    std::string getAdditionalBuildOptions() override {
        // Only handle material types that exist in scene
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        unsigned int typeBits = tracer->getScene()->getMaterialTypes();
        return getBxdfDefines(typeBits);
    }
};

class WFResetKernel : public clt::Kernel
{
public:
    WFResetKernel(void) : Kernel("src/wf_reset.cl", "reset") {}
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
        clt::check(err, "Failed to set wf_reset arguments!");
    }
};

class MKResetKernel : public clt::Kernel
{
public:
    MKResetKernel(void) : Kernel("src/mk_reset.cl", "reset") {}
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels", ctx->deviceBuffers.pixelBuffer);
        err |= setArg("denoiserAlbedo", ctx->deviceBuffers.denoiserAlbedoBuffer);
        err |= setArg("denoiserNormal", ctx->deviceBuffers.denoiserNormalBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        clt::check(err, "Failed to set mk_reset arguments!");
    }
};

class MKRaygenKernel : public clt::Kernel
{
public:
    MKRaygenKernel(void) : Kernel("src/mk_raygen.cl", "genCameraRays") {}
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        clt::check(err, "Failed to set mk_raygen arguments!");
    }
};

class MKNextVertexKernel : public clt::Kernel
{
public:
    MKNextVertexKernel(void) : Kernel("src/mk_next_vertex.cl", "nextVertex") {}
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
        clt::check(err, "Failed to set mk_next_vertex arguments!");
    }

    std::string getAdditionalBuildOptions() override {
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        const RenderParams& params = tracer->getParams();
        std::string opts;
        if (tracer->useDenoiser) opts.append(" -DUSE_OPTIX_DENOISER");
        return opts;
    }
};

class MKSampleBSDFKernel : public clt::Kernel
{
public:
    MKSampleBSDFKernel(void) : Kernel("src/mk_sample_bsdf.cl", "sampleBsdf") {}
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
        clt::check(err, "Failed to set mk_sample_bsdf arguments!");
    }

    std::string getAdditionalBuildOptions() override {
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        const RenderParams& params = tracer->getParams();
        std::string opts;
        if (tracer->useDenoiser) opts.append(" -DUSE_OPTIX_DENOISER");

        // Only check for material types that exist
        unsigned int typeBits = tracer->getScene()->getMaterialTypes();
        opts.append(getBxdfDefines(typeBits));

        return opts;
    }
};

class MKSplatKernel : public clt::Kernel
{
public:
    MKSplatKernel(void) : Kernel("src/mk_splat.cl", "splat") {}
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels", ctx->deviceBuffers.pixelBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("stats", ctx->deviceBuffers.renderStats);
        err |= setArg("numTasks", ctx->getNumTasks());
        clt::check(err, "Failed to set mk_splat arguments!");
    }
};

class MKSplatPreviewKernel : public clt::Kernel
{
public:
    MKSplatPreviewKernel(void) : Kernel("src/mk_splat_preview.cl", "splatPreview") {}
    void setArgs() override {
        CLContext *ctx = getCtxPtr(userPtr);
        int err = 0;
        err |= setArg("tasks", ctx->deviceBuffers.tasksBuffer);
        err |= setArg("pixels", ctx->deviceBuffers.pixelBuffer);
        err |= setArg("params", ctx->deviceBuffers.renderParams);
        err |= setArg("numTasks", ctx->getNumTasks());
        clt::check(err, "Failed to set mk_splat_preview arguments!");
    }
};

class MKPostprocessKernel : public clt::Kernel
{
public:
    MKPostprocessKernel(void) : Kernel("src/mk_postprocess.cl", "process") {}
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
        clt::check(err, "Failed to set mk_postprocess arguments!");
    }

    std::string getAdditionalBuildOptions() override {
        Tracer* tracer = static_cast<Tracer*>(userPtr);
        const RenderParams& params = tracer->getParams();
        std::string opts;
        if (tracer->useDenoiser) opts.append(" -DUSE_OPTIX_DENOISER");
        return opts;
    }
};
