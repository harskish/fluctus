#include "kernelreader.hpp"

// Needs to use custom preprocessor to avoid NVIDIA cache issues
void kernelFromSource(const std::string filename, cl::Context &context, cl::Program &program, int &err)
{
    std::string tmp = readKernel(filename);
    program = cl::Program(context, tmp, false, &err);
}

// Perform include expanding
void kernelFromSourceExpanded(const std::string filename, cl::Context & context, cl::Program & program, int & err)
{
    std::string expandedSrc = readKernel(filename);
    program = cl::Program(context, expandedSrc, false, &err);
}

void kernelFromBinary(const std::string filename, cl::Context & context, cl::Device & device, cl::Program & program, int & err)
{
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    std::cout << "Reding kernel binary " << filename << std::endl;

    if (!f)
    {
        std::cout << "Could not open kernel binary '" + filename + "'" << std::endl;
        waitExit();
    }

    std::ifstream::pos_type pos = f.tellg();
    cl::vector<unsigned char> binary(pos);
    f.seekg(0, std::ios::beg);
    f.read((char*)(&binary[0]), pos);

    cl::Program::Binaries binaries;
    binaries.push_back(binary);
    cl::vector<cl_int> status;
    cl::vector<cl::Device> devices = { device };
    program = cl::Program(context, devices, binaries, &status, &err);

    // Check compilation status
    for (cl_int i : status) {
        err |= i;
    }
}

void verify(const char* msg, int err)
{
    if (err != CL_SUCCESS)
    {
        std::cout << "ERROR: " << msg << std::endl;
        waitExit();
    }
}

// Checks kernel cache for match, otherwise loads from source
cl::Program kernelFromFile(const std::string filename, const std::string buildOpts, cl::Platform & platform, cl::Context & context, cl::Device & device, int & err)
{
    std::string sourcePath = "src/" + filename;

    // Compute hash of kernel source + build configuration
    std::string kernelSource = readKernel(sourcePath);

    // Separate binaries by (build options X platform name X device name)
    kernelSource += buildOpts;
    kernelSource += platform.getInfo<CL_PLATFORM_NAME>();
    kernelSource += device.getInfo<CL_DEVICE_NAME>();

    size_t hash = computeHash(kernelSource.data(), kernelSource.size());
    std::string binaryPath = "data/kernel_binaries/" + filename + "." + std::to_string(hash) + ".bin";

    cl::Program program;

    // Try to open cached kernel binary
    std::ifstream binaryFile(binaryPath, std::ios::binary | std::ios::ate);
    if (binaryFile)
    {
        std::cout << "Loading hashed kernel " << binaryPath << std::endl;

        std::ifstream::pos_type pos = binaryFile.tellg();
        cl::vector<unsigned char> binary(pos);
        binaryFile.seekg(0, std::ios::beg);
        binaryFile.read((char*)(&binary[0]), pos);

        cl::Program::Binaries binaries;
        binaries.push_back(binary);
        cl::vector<cl_int> status;
        cl::vector<cl::Device> devices = { device };
        program = cl::Program(context, devices, binaries, &status, &err);

        // Check program status
        for (cl_int i : status) err |= i;
        verify("Failed to create program from binary", err);

        // Build
        err = program.build(devices, buildOpts.c_str());
        verify("Failed to build program loaded from binary", err);
    }
    else
    {
        std::cout << "Building kernel " << filename << std::endl;

        kernelFromSource(sourcePath, context, program, err);
        cl::vector<cl::Device> devices = { device };
        err = program.build(devices, buildOpts.c_str());

        // Check build log
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        if (buildLog.length() > 2)
            std::cout << "\n[" << filename << " build log]:" << buildLog << std::endl;

        verify("Kernel compilation failed", err);
        
        auto ptxs = program.getInfo<CL_PROGRAM_BINARIES>();
        std::vector<unsigned char> ptx = ptxs[0];
        verify("Incorrect number of kernel binaries generated!", ptxs.size() != 1);

        // Open target file in overwrite-mode
        std::ofstream stream;
        stream.open(binaryPath, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        if (!stream.good())
        {
            std::cout << "Failed to open binary output file" << std::endl;
            waitExit();
        }

        // Write binary to file
        stream.write((const char*)ptx.data(), ptx.size());
        if (!stream.good())
        {
            std::cout << "Failed to write kernel binary" << std::endl;
            waitExit();
        }
        
        stream.close();
        std::cout << "Created cached kernel " << binaryPath << std::endl;
    }

    return program;
}

// Read kernel file, handle includes by recursion
// Used to enable kernel caching on NVIDIA hardware
std::string readKernel(std::string path, std::vector<std::string> &incl)
{
    path = unixifyPath(path);
    std::ifstream file(path);
    if (!file)
    {
        std::cout << "Cannot open file " << path << std::endl;
        waitExit();
    }

    // Don't include same file several times
    if (std::find(incl.begin(), incl.end(), path) != incl.end())
        return "";
    else
        incl.push_back(path);

    size_t idx = path.find_last_of('/');
    std::string dir = path.substr(0, idx);

    std::string output;
    std::string line;

    while (file.good())
    {
        getline(file, line);

        if (line.find("#include") == std::string::npos)
        {
            output.append(line + "\n");
        }
        else
        {
            std::string includeFileName = line.substr(10, line.length() - 11);
            if (endsWith(includeFileName, ".cl") || endsWith(includeFileName, ".h")) {
                std::string toAppend = readKernel(dir + "/" + includeFileName, incl);
                output.append(toAppend + "\n");
            }
        }
    }

    return output;
}
