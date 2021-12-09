#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <exception>
#include <string>
#include <sstream>
#include <math.h>

#define __CL_ENABLE_EXCEPTIONS

#include <CL/opencl.h>
#include <CL/cl.hpp>
#include <iostream>

int main(){
    try{

        //get all platforms (drivers)
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);
        if (all_platforms.size() == 0){
            std::cout << " No platforms found. Check OpenCL installation!\n";
            exit(1);
        }
        cl::Platform default_platform = all_platforms[0];
        std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

        //get default device of the default platform
        std::vector<cl::Device> all_devices;
        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if (all_devices.size() == 0){
            std::cout << " No devices found. Check OpenCL installation!\n";
            exit(1);
        }
        cl::Device default_device = all_devices[0];
        std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

        cl::Context context({ default_device });

        cl::Program::Sources sources;

        std::ifstream functions("src/functions.opencl");
        std::string functions_source((std::istreambuf_iterator<char>(functions)), (std::istreambuf_iterator<char>()));
        sources.push_back({functions_source.c_str(), functions_source.length()});

        cl::Program program(context, sources);
        try {
            program.build({ default_device });
        }
        catch (cl::Error err) {
            std::cout << " Error building: " << 
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
            exit(1);
        }

        // Read data file into single line
        std::ifstream data("data.txt");
        std::string line;
        std::getline(data, line);
        data.close();

        // Parse data
        std::stringstream line_stream(line);
        std::string buf;
        std::vector<int> numbers;
        int max_value = 0;
        while(std::getline(line_stream, buf, ',')) {
            int val = 0;
            std::stringstream(buf) >> val;
            numbers.push_back(val);
            if (val > max_value) {
                max_value = val;
            }
        }
        max_value += 1;
        std::cout << "Got size: " << numbers.size() << '\n';
        std::cout << "Max number: " << max_value << '\n';

        // GPU Side buffers
        cl::Buffer number_buffer(context, CL_MEM_READ_WRITE, sizeof(int) * numbers.size());
        cl::Buffer fuel_buffer(context, CL_MEM_READ_WRITE, sizeof(int) * numbers.size() * max_value);
        cl::Buffer sum_buffer(context, CL_MEM_READ_WRITE, sizeof(int) * max_value);
        cl::Buffer result_buffer(context, CL_MEM_READ_WRITE, sizeof(int));
        
        // CPU Side buffers
        // int* fuel_result = (int*)malloc(max_value * numbers.size() * sizeof(int));
        // int* sum_result = (int*)malloc(max_value * sizeof(int));
        int result = 0;

        // //create queue to which we will push commands for 	the device.
        cl::CommandQueue queue(context, default_device);

        // Copy numbers and zero out buffers
        const int zero[] = {0};
        queue.enqueueWriteBuffer(number_buffer, CL_TRUE, 0, sizeof(int) * numbers.size(), numbers.data());
        queue.enqueueFillBuffer(fuel_buffer, zero, 0, sizeof(int) * numbers.size() * max_value);
        queue.enqueueFillBuffer(sum_buffer, zero, 0, sizeof(int) * max_value);
        queue.enqueueFillBuffer(result_buffer, zero, 0, sizeof(int));

        cl::Kernel solver1(program, "calc_fuel1");
        solver1.setArg(0, number_buffer);
        solver1.setArg(1, numbers.size());
        solver1.setArg(2, fuel_buffer);
        queue.enqueueNDRangeKernel(solver1, cl::NullRange, cl::NDRange(numbers.size(), max_value), cl::NullRange);
        queue.finish();

        cl::Kernel sum_kernel(program, "sum_fuel");
        sum_kernel.setArg(0, numbers.size());
        sum_kernel.setArg(1, fuel_buffer);
        sum_kernel.setArg(2, sum_buffer);
        queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(max_value), cl::NullRange);
        queue.finish();

        cl::Kernel min_kernel(program, "min_fuel");
        min_kernel.setArg(0, max_value);
        min_kernel.setArg(1, sum_buffer);
        min_kernel.setArg(2, result_buffer);
        queue.enqueueNDRangeKernel(min_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);

        queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0, sizeof(int), &result);
        queue.finish();

        std::cout << "Solution1: " << result << '\n';

        cl::Kernel solver2(program, "calc_fuel2");
        solver2.setArg(0, number_buffer);
        solver2.setArg(1, numbers.size());
        solver2.setArg(2, fuel_buffer);
        queue.enqueueNDRangeKernel(solver2, cl::NullRange, cl::NDRange(numbers.size(), max_value), cl::NullRange);
        queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(max_value), cl::NullRange);
        queue.enqueueNDRangeKernel(min_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
        queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0, sizeof(int), &result);
        queue.finish();

        std::cout << "Solution2: " << result << '\n';

        std::cout << std::endl;
    }
    catch (cl::Error err)
    {
        printf("Error: %s (%d)\n", err.what(), err.err());
        return -1;
    }
    
    printf("Closing..");

    return 0;
}	