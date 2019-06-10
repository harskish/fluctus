/*  The MIT License (MIT)
 *
 *  Copyright (c) 2019 Erik Härkönen
 *  Copyright (c) 2019 Matias Koskela / Tampere University
 *  Copyright (c) 2018 Kalle Immonen / Tampere University of Technology
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include "BMFRDenoiser.hpp"
#include "tracer.hpp"
#include "clcontext.hpp"
#include "window.hpp"
#include <clt.hpp>
#include <sstream>

//#include "OpenImageIO/imageio.h" TODO: use TinyEXR instead!
#include "bmfr/CLUtils/CLUtils.hpp"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "IL/ilu.h"

#define _CRT_SECURE_NO_WARNINGS
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// ### Choose your OpenCL device and platform with these defines ###
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0


// ### Edit these defines if you have different input ###
// TODO detect IMAGE_SIZES automatically from the input files
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720
// TODO detect FRAME_COUNT from the input files
#define FRAME_COUNT 60
// Location where input frames and feature buffers are located
#define INPUT_DATA_PATH sponza-(glossy)/inputs
//#define INPUT_DATA_PATH san_miguel/inputs
#define INPUT_DATA_PATH_STR STR(INPUT_DATA_PATH)

// camera_matrices.h is expected to be in the same folder
//#include STR(INPUT_DATA_PATH/camera_matrices.h)

//const float position_limit_squared = 0.001600;
//const float normal_limit_squared = 0.250000;

// Every matrix is multiplication of camera projection, rotation and position matrices
//const float camera_matrices[1][4][4] = {
//   { // frame 0
//      {0.821398, 0.958759, -0.862917, -0.862916},
//      {0.000000, 3.290975, 0.305999, 0.305999},
//      {-1.762433, 0.446838, -0.402170, -0.402170},
//      {-2.446395, -10.698995, -0.135571, -0.065571},
//   },
//};
//const float pixel_offsets[1][2] = {
//   {0.500000, 0.500000}, // frame 0
//};
#include STR(../INPUT_DATA_PATH/camera_matrices.h)


// These names are appended with NN.exr, where NN is the frame number
#define NOISY_FILE_NAME INPUT_DATA_PATH_STR"/color"
#define NORMAL_FILE_NAME INPUT_DATA_PATH_STR"/shading_normal"
#define POSITION_FILE_NAME INPUT_DATA_PATH_STR"/world_position"
#define ALBEDO_FILE_NAME INPUT_DATA_PATH_STR"/albedo"
#define OUTPUT_FILE_NAME "outputs/output"


// ### Edit these defines if you want to experiment different parameters ###
// The amount of noise added to feature buffers to cancel sigularities
#define NOISE_AMOUNT 1e-2
// The amount of new frame used in accumulated frame (1.f would mean no accumulation).
#define BLEND_ALPHA 0.2f
#define SECOND_BLEND_ALPHA 0.1f
#define TAA_BLEND_ALPHA 0.2f
// NOTE: if you want to use other than normal and world_position data you have to make
// it available in the first accumulation kernel and in the weighted sum kernel
#define NOT_SCALED_FEATURE_BUFFERS \
"1.f,"\
"normal.x,"\
"normal.y,"\
"normal.z,"\
// The next features are not in the range from -1 to 1 so they are scaled to be from 0 to 1.
#define SCALED_FEATURE_BUFFERS \
"world_position.x,"\
"world_position.y,"\
"world_position.z,"\
"world_position.x*world_position.x,"\
"world_position.y*world_position.y,"\
"world_position.z*world_position.z"


// ### Edit these defines to change optimizations for your target hardware ###

// If 1 uses ~half local memory space for R, but computing indexes is more complicated
#define COMPRESSED_R 1

// If 1 stores tmp_data to private memory when it is loaded for dot product calculation
#define CACHE_TMP_DATA 1

// If 1 tmp_data buffer is in half precision for faster load and store.
// NOTE: if world position values are greater than 256 this cannot be used because
// 256*256 is infinity in half-precision
#define USE_HALF_PRECISION_IN_TMP_DATA 1

// If 1 adds __attribute__((reqd_work_group_size(256, 1, 1))) to fitter and
// accumulate_noisy_data kernels. With some codes, attribute made the kernels faster and
// with some it slowed them down.
#define ADD_REQD_WG_SIZE 1

// These local sizes are used with 2D kernels which do not require spesific local size
// (Global sizes are always a multiple of 32)
#define LOCAL_WIDTH 32
#define LOCAL_HEIGHT 1
// Fastest on AMD Radeon Vega Frontier Edition was (LOCAL_WIDTH = 256, LOCAL_HEIGHT = 1)
// Fastest on Nvidia Titan Xp was (LOCAL_WIDTH = 32, LOCAL_HEIGHT = 1)



// ### Do not edit defines after this line unless you know what you are doing ###
// For example, other than 32x32 blocks are not supported
#define BLOCK_EDGE_LENGTH 32
#define BLOCK_PIXELS (BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH)
// Rounds image sizes up to next multiple of BLOCK_EDGE_LENGTH
#define WORKSET_WIDTH (BLOCK_EDGE_LENGTH * \
   ((IMAGE_WIDTH + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))
#define WORKSET_HEIGHT (BLOCK_EDGE_LENGTH * \
   ((IMAGE_HEIGHT + BLOCK_EDGE_LENGTH - 1) / BLOCK_EDGE_LENGTH))
#define WORKSET_WITH_MARGINS_WIDTH (WORKSET_WIDTH + BLOCK_EDGE_LENGTH)
#define WORKSET_WITH_MARGINS_HEIGHT (WORKSET_HEIGHT + BLOCK_EDGE_LENGTH)
#define OUTPUT_SIZE (WORKSET_WIDTH * WORKSET_HEIGHT)
// 256 is the maximum local size on AMD GCN
// Synchronization within 32x32=1024 block requires unrollign four times
#define LOCAL_SIZE 256
#define FITTER_GLOBAL (LOCAL_SIZE * ((WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH) * \
   (WORKSET_WITH_MARGINS_HEIGHT / BLOCK_EDGE_LENGTH)))

struct Operation_result
{
    bool success;
    std::string error_message;
    Operation_result(bool success, const std::string &error_message = "") :
        success(success), error_message(error_message) {}
};


Operation_result read_image_file(
    const std::string& file_name, const int frame, float* buffer)
{
    float* out; // width * height * RGBA
    int width;
    int height;
    const char* err = nullptr;

    std::string input = file_name + std::to_string(frame) + ".exr";
    int ret = LoadEXR(&out, &width, &height, input.c_str(), &err);

    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "ERR : %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }

        return { false, "Can't open image: " + file_name };
    }
    else {
        if (width != IMAGE_WIDTH || height != IMAGE_HEIGHT) {
            free(out);
            return { false, "Dimensions don't match for " + file_name };
        }

        // Convert RGBA to RGB
        for (int i = 0; i < width * height; i++) {
            buffer[i * 3 + 0] = out[i * 4 + 0];
            buffer[i * 3 + 1] = out[i * 4 + 1];
            buffer[i * 3 + 2] = out[i * 4 + 2];
        }
    }

    free(out);
    return { true };
}

Operation_result load_image(cl_float* image, const std::string file_name, const int frame)
{
    Operation_result result = read_image_file(file_name, frame, image);

    if (!result.success)
        return result;

    return { true };
}


// TODO: Move to BMFRKernels.hpp?

inline BMFRDenoiser* getDenoiserPtr(void* userPtr)
{
    Tracer* tracer = static_cast<Tracer*>(userPtr);
    BMFRDenoiser* denoiser = static_cast<BMFRDenoiser*>(tracer->getDenoiser().get());
    return denoiser;
}

class BMFRKernelBase : public clt::Kernel
{
protected:
    BMFRKernelBase(const char* entryPoint) : Kernel("src/bmfr/bmfr.cl", entryPoint) {}
    std::string getAdditionalBuildOptions() override {
        BMFRDenoiser* denoiser = getDenoiserPtr(userPtr);
        std::stringstream buildOptions;
        buildOptions <<
            " -D BUFFER_COUNT=" << denoiser->buffer_count <<
            " -D FEATURES_NOT_SCALED=" << denoiser->features_not_scaled_count <<
            " -D FEATURES_SCALED=" << denoiser->features_scaled_count <<
            " -D IMAGE_WIDTH=" << IMAGE_WIDTH <<
            " -D IMAGE_HEIGHT=" << IMAGE_HEIGHT <<
            " -D WORKSET_WIDTH=" << WORKSET_WIDTH <<
            " -D WORKSET_HEIGHT=" << WORKSET_HEIGHT <<
            " -D FEATURE_BUFFERS=" << NOT_SCALED_FEATURE_BUFFERS SCALED_FEATURE_BUFFERS <<
            " -D LOCAL_WIDTH=" << LOCAL_WIDTH <<
            " -D LOCAL_HEIGHT=" << LOCAL_HEIGHT <<
            " -D WORKSET_WITH_MARGINS_WIDTH=" << WORKSET_WITH_MARGINS_WIDTH <<
            " -D WORKSET_WITH_MARGINS_HEIGHT=" << WORKSET_WITH_MARGINS_HEIGHT <<
            " -D BLOCK_EDGE_LENGTH=" << STR(BLOCK_EDGE_LENGTH) <<
            " -D BLOCK_PIXELS=" << BLOCK_PIXELS <<
            " -D R_EDGE=" << denoiser->buffer_count - 2 <<
            " -D NOISE_AMOUNT=" << STR(NOISE_AMOUNT) <<
            " -D BLEND_ALPHA=" << STR(BLEND_ALPHA) <<
            " -D SECOND_BLEND_ALPHA=" << STR(SECOND_BLEND_ALPHA) <<
            " -D TAA_BLEND_ALPHA=" << STR(TAA_BLEND_ALPHA) <<
            " -D POSITION_LIMIT_SQUARED=" << position_limit_squared <<
            " -D NORMAL_LIMIT_SQUARED=" << normal_limit_squared <<
            " -D COMPRESSED_R=" << STR(COMPRESSED_R) <<
            " -D CACHE_TMP_DATA=" << STR(CACHE_TMP_DATA) <<
            " -D ADD_REQD_WG_SIZE=" << STR(ADD_REQD_WG_SIZE) <<
            " -D LOCAL_SIZE=" << STR(LOCAL_SIZE) <<
            " -D USE_HALF_PRECISION_IN_TMP_DATA=" << STR(USE_HALF_PRECISION_IN_TMP_DATA);

        return buildOptions.str();
    }
};

class FitterKernel : public BMFRKernelBase
{
public:
    FitterKernel(void) : BMFRKernelBase("fitter") {}
    void setArgs() override
    {
        BMFRDenoiser* denoiser = getDenoiserPtr(userPtr);
#if COMPRESSED_R
        const int r_size = ((denoiser->buffer_count - 2) *
            (denoiser->buffer_count - 1) / 2) *
            sizeof(cl_float3);
#else
        const int r_size = (buffer_count - 2) *
            (buffer_count - 2) * sizeof(cl_float3);
#endif

        int err = 0;
        err |= setArg("sum_vec", LOCAL_SIZE * sizeof(float), nullptr);
        err |= setArg("u_vec", BLOCK_PIXELS * sizeof(float), nullptr);
        err |= setArg("r_mat", r_size, nullptr);
        err |= setArg("weights", denoiser->weights_buffer);
        err |= setArg("mins_maxs", denoiser->mins_maxs_buffer);
        err |= setArg("frame_number", denoiser->frame);
        clt::check(err, "Failed to set BMFR fitter arguments");
    }
};

class WeightedSumKernel : public BMFRKernelBase
{
public:
    WeightedSumKernel(void) : BMFRKernelBase("weighted_sum") {}
    void setArgs() override
    {
        BMFRDenoiser* denoiser = getDenoiserPtr(userPtr);
        int err = 0;
        err |= setArg("weights", denoiser->weights_buffer);
        err |= setArg("mins_maxs", denoiser->mins_maxs_buffer);
        err |= setArg("output", denoiser->filtered_buffer);
        clt::check(err, "Failed to set BMFR weighted_sum arguments");
    }
};

class AccumNoisyKernel : public BMFRKernelBase
{
public:
    AccumNoisyKernel(void) : BMFRKernelBase("accumulate_noisy_data") {}
    void setArgs() override
    {
        BMFRDenoiser* denoiser = getDenoiserPtr(userPtr);
        int err = 0;
        err |= setArg("out_prev_frame_pixel", denoiser->prev_pixels_buffer);
        err |= setArg("accept_bools", denoiser->accept_buffer);
        clt::check(err, "Failed to set BMFR accumulate_noisy_data arguments");
    }
};

class AccumFilteredKernel : public BMFRKernelBase
{
public:
    AccumFilteredKernel(void) : BMFRKernelBase("accumulate_filtered_data") {}
    void setArgs() override
    {
        BMFRDenoiser* denoiser = getDenoiserPtr(userPtr);
        int err = 0;
        err |= setArg("filtered_frame", denoiser->filtered_buffer);
        err |= setArg("in_prev_frame_pixel", denoiser->prev_pixels_buffer);
        err |= setArg("accept_bools", denoiser->accept_buffer);
        err |= setArg("albedo", denoiser->albedo_buffer);
        err |= setArg("tone_mapped_frame", denoiser->tone_mapped_buffer);
        clt::check(err, "Failed to set BMFR accumulate_filtered_data arguments");
    }
};

class TAAKernel : public BMFRKernelBase
{
public:
    TAAKernel(void) : BMFRKernelBase("taa") {}
    void setArgs() override
    {
        BMFRDenoiser* denoiser = getDenoiserPtr(userPtr);
        int err = 0;
        err |= setArg("in_prev_frame_pixel", denoiser->prev_pixels_buffer);
        err |= setArg("new_frame", denoiser->tone_mapped_buffer);
        err |= setArg("result_frame", *denoiser->result_buffer.current());
        err |= setArg("prev_frame", *denoiser->result_buffer.previous());
        err |= setArg("frame_number", denoiser->frame);
        clt::check(err, "Failed to set BMFR taa arguments");
    }
};

void BMFRDenoiser::setup(CLContext* context, PTWindow* window)
{
    printf("BMFR initialize.\n");
    ctx = context;

    std::string features_not_scaled(NOT_SCALED_FEATURE_BUFFERS);
    std::string features_scaled(SCALED_FEATURE_BUFFERS);
    features_not_scaled_count =
        std::count(features_not_scaled.begin(), features_not_scaled.end(), ',');
    // + 1 because last one does not have ',' after it.
    features_scaled_count =
        std::count(features_scaled.begin(), features_scaled.end(), ',') + 1;

    // + 3 stands for three noisy channels.
    buffer_count = features_not_scaled_count + features_scaled_count + 3;

    clt::State& state = ctx->getState();

    // Create buffers
    normals_buffer = Double_buffer<cl::Buffer>(state.context,
        CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));
    positions_buffer = Double_buffer<cl::Buffer>(state.context,
        CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));
    noisy_buffer = Double_buffer<cl::Buffer>(state.context,
        CL_MEM_READ_WRITE, OUTPUT_SIZE * 3 * sizeof(cl_float));
    size_t in_buffer_data_size = USE_HALF_PRECISION_IN_TMP_DATA ? sizeof(cl_half) : sizeof(cl_float);
    in_buffer = cl::Buffer(state.context, CL_MEM_READ_WRITE, WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT *
        buffer_count * in_buffer_data_size, nullptr);
    filtered_buffer = cl::Buffer(state.context, CL_MEM_READ_WRITE,
        OUTPUT_SIZE * 3 * sizeof(cl_float));
    out_buffer = Double_buffer<cl::Buffer>(state.context, CL_MEM_READ_WRITE,
        WORKSET_WITH_MARGINS_WIDTH * WORKSET_WITH_MARGINS_HEIGHT * 3 * sizeof(cl_float));
    result_buffer = Double_buffer<cl::Buffer>(state.context, CL_MEM_READ_WRITE,
        OUTPUT_SIZE * 3 * sizeof(cl_float));
    prev_pixels_buffer = cl::Buffer(state.context, CL_MEM_READ_WRITE,
        OUTPUT_SIZE * sizeof(cl_float2));
    accept_buffer = cl::Buffer(state.context, CL_MEM_READ_WRITE, OUTPUT_SIZE * sizeof(cl_uchar));
    albedo_buffer = cl::Buffer(state.context, CL_MEM_READ_ONLY,
        IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float));
    tone_mapped_buffer = cl::Buffer(state.context, CL_MEM_READ_WRITE,
        IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(cl_float));
    weights_buffer = cl::Buffer(state.context, CL_MEM_READ_WRITE,
        (FITTER_GLOBAL / 256) * (buffer_count - 3) * 3 * sizeof(cl_float));
    mins_maxs_buffer = cl::Buffer(state.context, CL_MEM_READ_WRITE,
        (FITTER_GLOBAL / 256) * 6 * sizeof(cl_float2));
    spp_buffer = Double_buffer<cl::Buffer>(state.context, CL_MEM_READ_WRITE,
        OUTPUT_SIZE * sizeof(cl_char));

    all_double_buffers = std::vector<Double_buffer<cl::Buffer> *> {
        &normals_buffer, &positions_buffer, &noisy_buffer,
        &out_buffer, &result_buffer, &spp_buffer };


    // Kernels
    window->showMessage("Building kernel", "BMFR fitter");
    fitter_kernel = new FitterKernel();
    fitter_kernel->build(state.context, state.device, state.platform);

    window->showMessage("Building kernel", "BMFR weighted sum");
    weighted_sum_kernel = new WeightedSumKernel();
    weighted_sum_kernel->build(state.context, state.device, state.platform);

    window->showMessage("Building kernel", "BMFR accum noisy");
    accum_noisy_kernel = new AccumNoisyKernel();
    accum_noisy_kernel->build(state.context, state.device, state.platform);

    window->showMessage("Building kernel", "BMFR accum filtered");
    accum_filtered_kernel = new AccumFilteredKernel();
    accum_filtered_kernel->build(state.context, state.device, state.platform);

    window->showMessage("Building kernel", "BMFR TAA");
    taa_kernel = new TAAKernel();
    taa_kernel->build(state.context, state.device, state.platform);

    window->hideMessage();
}

void BMFRDenoiser::bindBuffers(PTWindow *window)
{

}

void BMFRDenoiser::resizeBuffers(PTWindow *window)
{

}

void BMFRDenoiser::denoise()
{
    cl::NDRange accum_global(WORKSET_WITH_MARGINS_WIDTH, WORKSET_WITH_MARGINS_HEIGHT);
    cl::NDRange output_global(WORKSET_WIDTH, WORKSET_HEIGHT);
    cl::NDRange local(LOCAL_WIDTH, LOCAL_HEIGHT);
    cl::NDRange fitter_global(FITTER_GLOBAL);
    cl::NDRange fitter_local(LOCAL_SIZE);

    cl::CommandQueue& queue = ctx->getState().cmdQueue;

    // Data arrays
    std::vector<cl_float> out_data;
    std::vector<cl_float> albedos;
    std::vector<cl_float> normals;
    std::vector<cl_float> positions;
    std::vector<cl_float> noisy_input;

    {
        out_data.resize(3 * OUTPUT_SIZE);

        albedos.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        Operation_result result = load_image(albedos.data(), ALBEDO_FILE_NAME,
            frame);

        bool error = false;
        if (!result.success)
        {
            error = true;
            printf("Albedo buffer loading failed, reason: %s\n",
                result.error_message.c_str());
        }

        normals.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(normals.data(), NORMAL_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Normal buffer loading failed, reason: %s\n",
                result.error_message.c_str());
        }

        positions.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(positions.data(), POSITION_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Position buffer loading failed, reason: %s\n",
                result.error_message.c_str());
        }

        noisy_input.resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(noisy_input.data(), NOISY_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Position buffer loading failed, reason: %s\n",
                result.error_message.c_str());
        }

        if (error) {
            throw std::runtime_error("Could not load input exrs");
        }
    }

    int err = 0;

    err |= queue.enqueueWriteBuffer(albedo_buffer, true, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 3 *
        sizeof(cl_float), albedos.data());
    err |= queue.enqueueWriteBuffer(*normals_buffer.current(), true, 0, IMAGE_WIDTH *
        IMAGE_HEIGHT * 3 * sizeof(cl_float), normals.data());
    err |= queue.enqueueWriteBuffer(*positions_buffer.current(), true, 0, IMAGE_WIDTH *
        IMAGE_HEIGHT * 3 * sizeof(cl_float), positions.data());
    err |= queue.enqueueWriteBuffer(*noisy_buffer.current(), true, 0, IMAGE_WIDTH * IMAGE_HEIGHT *
        3 * sizeof(cl_float), noisy_input.data());


    // On the first frame accum_noisy_kernel just copies to the in_buffer
    err |= accum_noisy_kernel->setArg("current_normals", *normals_buffer.current());
    err |= accum_noisy_kernel->setArg("previous_normals", *normals_buffer.previous());
    err |= accum_noisy_kernel->setArg("current_positions", *positions_buffer.current());
    err |= accum_noisy_kernel->setArg("previous_positions", *positions_buffer.previous());
    err |= accum_noisy_kernel->setArg("current_noisy", *noisy_buffer.current());
    err |= accum_noisy_kernel->setArg("previous_noisy", *noisy_buffer.previous());
    err |= accum_noisy_kernel->setArg("previous_spp", *spp_buffer.previous());
    err |= accum_noisy_kernel->setArg("current_spp", *spp_buffer.current());
    err |= accum_noisy_kernel->setArg("tmp_data", in_buffer);
    const int matrix_index = frame == 0 ? 0 : frame - 1;
    err |= accum_noisy_kernel->setArg("prev_frame_camera_matrix", sizeof(cl_float16),
        &(camera_matrices[matrix_index][0][0]));
    err |= accum_noisy_kernel->setArg("pixel_offset", sizeof(cl_float2),
        &(pixel_offsets[frame][0]));
    err |= accum_noisy_kernel->setArg("frame_number", frame);
    err |= queue.enqueueNDRangeKernel(*accum_noisy_kernel, cl::NullRange, accum_global, local,
        nullptr); //, &accum_noisy_timer[matrix_index].event());

    err |= fitter_kernel->setArg("tmp_data", in_buffer);
    err |= fitter_kernel->setArg("frame_number", frame);
    err |= queue.enqueueNDRangeKernel(*fitter_kernel, cl::NullRange, fitter_global,
        fitter_local, nullptr); // , & fitter_timer[frame].event());

    //arg_index = 3;
    err |= weighted_sum_kernel->setArg("current_normals", *normals_buffer.current());
    err |= weighted_sum_kernel->setArg("current_positions", *positions_buffer.current());
    err |= weighted_sum_kernel->setArg("current_noisy", *noisy_buffer.current());
    err |= weighted_sum_kernel->setArg("frame_number", frame);
    err |= queue.enqueueNDRangeKernel(*weighted_sum_kernel, cl::NullRange, output_global,
        local, nullptr); // , & weighted_sum_timer[frame].event());

    //arg_index = 5;
    err |= accum_filtered_kernel->setArg("current_spp", *spp_buffer.current());
    err |= accum_filtered_kernel->setArg("accumulated_prev_frame", *out_buffer.previous());
    err |= accum_filtered_kernel->setArg("accumulated_frame", *out_buffer.current());
    err |= accum_filtered_kernel->setArg("frame_number", frame);
    err |= queue.enqueueNDRangeKernel(*accum_filtered_kernel, cl::NullRange, output_global,
        local, nullptr); // , & accum_filtered_timer[matrix_index].event());

    //arg_index = 2;
    err |= taa_kernel->setArg("result_frame", *result_buffer.current());
    err |= taa_kernel->setArg("prev_frame", *result_buffer.previous());
    err |= taa_kernel->setArg("frame_number", frame);
    err |= queue.enqueueAcquireGLObjects(&ctx->sharedMemory);
    err |= taa_kernel->setArg("preview_frame", ctx->deviceBuffers.previewBuffer); // FLUCTUS TEST
    err |= queue.enqueueNDRangeKernel(*taa_kernel, cl::NullRange, output_global, local,
        nullptr); // , & taa_timer[matrix_index].event());
    err |= queue.enqueueReleaseGLObjects(&ctx->sharedMemory);
    
    err |= queue.enqueueReadBuffer(*result_buffer.current(), false, 0,
        OUTPUT_SIZE * 3 * sizeof(cl_float), out_data.data());

    err |= queue.finish();
    clt::check(err, "BMFR denoising error!");

    static bool saveToDisk = false;
    if (saveToDisk)
    {
        unsigned int numBytes = IMAGE_WIDTH * IMAGE_HEIGHT * 3; // rgb
        std::unique_ptr<unsigned char[]> dataBytes(new unsigned char[numBytes]);

        // Convert to bytes
        // Already tonemapped and gamma-corrected
        int counter = 0;
        for (int i = 0; i < numBytes; i += 3)
        {
            float r = out_data[numBytes - 1 - (i + 2)];
            float g = out_data[numBytes - 1 - (i + 1)];
            float b = out_data[numBytes - 1 - (i + 0)];

            // Convert to bytes
            auto clamp = [](float value) { return std::max(0.0f, std::min(1.0f, value)); };
            dataBytes[counter++] = (unsigned char)(255 * clamp(r));
            dataBytes[counter++] = (unsigned char)(255 * clamp(g));
            dataBytes[counter++] = (unsigned char)(255 * clamp(b));
        }

        std::string outname = "bmfr_out/bmfr_" + std::to_string(frame) + ".png";
        ILuint imageID = ilGenImage();
        ilBindImage(imageID);
        ilTexImage(IMAGE_WIDTH, IMAGE_HEIGHT, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, dataBytes.get());
        ilSaveImage(outname.c_str());
        ilDeleteImage(imageID);
    }

    // Swap all double buffers
    std::for_each(all_double_buffers.begin(), all_double_buffers.end(),
        std::bind(&Double_buffer<cl::Buffer>::swap, std::placeholders::_1));
    
    frame++;
    if (frame >= 60) {
        saveToDisk = false;
        frame = 0;
    }
}

void BMFRDenoiser::setBlend(float val)
{
    std::cout << "Blend not supported for BMFR" << std::endl;
}
