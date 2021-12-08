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

        
        std::string kernel_code =
            "__kernel void calc_fuel(__global int* numbers, int numbers_size, __global int* fuel_buffer) {"
            "	int index = get_global_id(0);"
            "	int position = get_global_id(1);"
            // Solution to first problem
            "	fuel_buffer[position * numbers_size + index] = abs(numbers[index] - position);"
            // Solution to second problem
            //"	fuel_buffer[position * numbers_size + index] = (abs(numbers[index] - position)+1) * abs(numbers[index] - position) / 2;"
            "};";
        sources.push_back({ kernel_code.c_str(), kernel_code.length() });

        std::string sum_code =
            "__kernel void sum_fuel(int numbers_size, __global int* fuel_buffer, __global int* sum_buffer) {"
            "	int position = get_global_id(0);"
            "	int total = 0;"
            "   for (int i = 0; i < numbers_size; i++) {"
            "       total += fuel_buffer[position * numbers_size + i];"
            "   }"
            "   sum_buffer[position] = total;"
            "};";
        sources.push_back({ sum_code.c_str(), sum_code.length() });

        std::string min_code =
            "__kernel void min_fuel(int positions, __global int* sum_buffer, __global int* result_buffer) {"
            "	int total = 0;"
            "   int min_pos = 0;"
            "   for (int i = 0; i < positions; i++) {"
            "       if (sum_buffer[i] < sum_buffer[min_pos]) {"
            "           min_pos = i;"
            "       }"
            "   }"
            "   *result_buffer = sum_buffer[min_pos];"
            "};";
        sources.push_back({ min_code.c_str(), min_code.length() });

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

        cl::Buffer number_buffer(context, CL_MEM_READ_WRITE, sizeof(int) * numbers.size());
        cl::Buffer fuel_buffer(context, CL_MEM_READ_WRITE, sizeof(int) * numbers.size() * max_value);
        cl::Buffer sum_buffer(context, CL_MEM_READ_WRITE, sizeof(int) * max_value);
        cl::Buffer result_buffer(context, CL_MEM_READ_WRITE, sizeof(int));

        // //create queue to which we will push commands for 	the device.
        cl::CommandQueue queue(context, default_device);

        // //write arrays A and B to the device
        
        const int zero[] = {0};
        queue.enqueueWriteBuffer(number_buffer, CL_TRUE, 0, sizeof(int) * numbers.size(), numbers.data());
        queue.enqueueFillBuffer(fuel_buffer, zero, 0, sizeof(int) * numbers.size() * max_value);
        queue.enqueueFillBuffer(sum_buffer, zero, 0, sizeof(int) * max_value);
        queue.enqueueFillBuffer(result_buffer, zero, 0, sizeof(int));

        cl::Kernel kernel(program, "calc_fuel");

        kernel.setArg(0, number_buffer);
        kernel.setArg(1, numbers.size());
        kernel.setArg(2, fuel_buffer);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numbers.size(), max_value), cl::NullRange);

        //std::vector<std::vector<int>> fuel_result(max_value, std::vector<int>(numbers.size()));
        int* fuel_result = (int*)malloc(max_value * numbers.size() * sizeof(int));
        // //read result C from the device to array C
        queue.enqueueReadBuffer(fuel_buffer, CL_TRUE, 0, sizeof(int) * numbers.size() * max_value, fuel_result);
        queue.finish();
        
        // std::cout << "fuel result: \n";
        // for (int y = 0; y < max_value; y++){
        //     for (int x = 0; x < numbers.size(); x++){
        // 	    std::cout << fuel_result[y * numbers.size() + x] << " ";
        //     }
        //     std::cout << '\n';
        // }

        cl::Kernel sum_kernel(program, "sum_fuel");

        sum_kernel.setArg(0, numbers.size());
        sum_kernel.setArg(1, fuel_buffer);
        sum_kernel.setArg(2, sum_buffer);
        queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(max_value), cl::NullRange);

        int* sum_result = (int*)malloc(max_value * sizeof(int));
        queue.enqueueReadBuffer(sum_buffer, CL_TRUE, 0, sizeof(int) * max_value, sum_result);
        queue.finish();

        // std::cout << "sum result: \n";
        // for (int y = 0; y < max_value; y++){
        // 	std::cout << sum_result[y] << " ";
        //     std::cout << '\n';
        // }

        cl::Kernel min_kernel(program, "min_fuel");

        min_kernel.setArg(0, max_value);
        min_kernel.setArg(1, sum_buffer);
        min_kernel.setArg(2, result_buffer);
        queue.enqueueNDRangeKernel(min_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);

        int* result = (int*)malloc(sizeof(int));
        queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0, sizeof(int), result);
        queue.finish();

        std::cout << "Solution1: " << *result << '\n';

        free(result);
        free(sum_result);
        free(fuel_result);

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