#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (16384)

void print_ocl_time(cl_event event, cl_command_queue queue, char* event_name) {
    clWaitForEvents(1, &event);
    clFinish(queue);
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);

    unsigned long duration = time_end-time_start;
    double duration_s = (double) duration / 1e9;
    printf("`%s` time is: %.4f milliseconds \n", event_name, duration_s);
    clReleaseEvent(event);
}

int ocl_init(char * kernel_filename, cl_command_queue * command_queue, cl_context * context,
            cl_program * program) 
{
    int i;
    cl_int clStatus=0;

    // Read kernel from file
    FILE *fp;
    char *source_str;
    size_t source_size;

    printf("Compiling kernels in `%s`\n", kernel_filename);
    fp = fopen(kernel_filename, "r");
    if (!fp) {
        fprintf(stderr, "Something went wrong - can't find file %s\n", kernel_filename);
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        source_str[source_size] = '\0';
    fclose( fp );

    // Get platforms
    cl_uint num_platforms;
    clStatus |= clGetPlatformIDs(0, NULL, &num_platforms);
    
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus |= clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get platform devices
    cl_uint num_devices;
    clStatus |= clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus |= clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("Error while reading devices etc., code %d\n", clStatus);
        return 1;
    }


    // Context
    *context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("Error while creating context, return value %d\n", clStatus);
        return 1;
    }

 
    // Command queue
    *command_queue = clCreateCommandQueue(*context, devices[0],
            CL_QUEUE_PROFILING_ENABLE, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("Error while creating command queue, return value %d\n", clStatus);
        return 1;
    }

    // Create and build a program
    *program = clCreateProgramWithSource(*context, 1, (const char **)&source_str, NULL, &clStatus);
    clStatus = clBuildProgram(*program, 1, devices, NULL, NULL, NULL);
    size_t build_log_len;
    clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);

    // Log
    if (build_log_len > 2)
    {
        char *build_log;
        build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
        clStatus = clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, 
                                        build_log_len, build_log, NULL);
        printf("%s\n", build_log);
        free(build_log);
        return 1;
    }
    if (clStatus != CL_SUCCESS) {
        printf("Error while creating program, return value %d\n", clStatus);
        return 1;
    }
    
    free(source_str);
    free(devices);
    free(platforms);
    return 0;
}

int ocl_destroy(cl_command_queue command_queue, cl_context context,
            cl_program program) {
    // release & free
    cl_int clStatus;
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
}

void ocl_swap_pointers(cl_mem * a, cl_mem * b) {
    cl_mem c = *a;
    *a = *b;
    *b = c;
}