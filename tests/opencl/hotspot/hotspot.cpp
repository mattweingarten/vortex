#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// Define Vortex-specific parameters
#define BLOCK_SIZE 8
#define GRID_COLS 8
#define GRID_ROWS 8

// Simple timing functions
void start_timer(struct timeval *start) {
    gettimeofday(start, NULL);
}

float stop_timer(struct timeval start) {
    struct timeval end;
    gettimeofday(&end, NULL);
    return ((end.tv_sec - start.tv_sec) * 1000.0) + 
           ((end.tv_usec - start.tv_usec) / 1000.0);
}

// Utility function to read kernel file
char* readKernelFile(const char* filename, size_t* size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Failed to open kernel file: %s\n", filename);
        exit(1);
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    rewind(fp);
    
    char* source = (char*)malloc(*size + 1);
    if (!source) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    
    size_t bytes_read = fread(source, 1, *size, fp);
    if (bytes_read != *size) {
        printf("Error reading file: only read %zu of %zu bytes\n", bytes_read, *size);
        free(source);
        exit(1);
    }
    
    source[*size] = '\0';
    fclose(fp);
    
    return source;
}

int main() {
    struct timeval start;
    start_timer(&start);
    
    // Host data
    float *h_temp, *h_power, *h_result;
    int grid_rows = GRID_ROWS;
    int grid_cols = GRID_COLS;
    int size = grid_rows * grid_cols;
    
    // Hotspot algorithm parameters
    int iteration = 1;      // Number of iterations
    int border_cols = 1;    // Border offset
    int border_rows = 1;    // Border offset
    float Cap = 0.5;        // Capacitance
    float Rx = 2.0;         // Thermal resistances
    float Ry = 2.0;
    float Rz = 2.0;
    float step = 0.5;       // Time step
    
    // Allocate host memory
    h_power = (float*)malloc(size * sizeof(float));
    h_temp = (float*)malloc(size * sizeof(float));
    h_result = (float*)malloc(size * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_temp[i] = 80.0;
        h_power[i] = 0.7;
    }
    
    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_power, d_temp_src, d_temp_dst;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting platform ID: %d\n", err);
        return -1;
    }
    
    // Get device (Vortex device)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting device ID: %d\n", err);
        return -1;
    }
    
    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating context: %d\n", err);
        return -1;
    }
    
    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating command queue: %d\n", err);
        return -1;
    }
    
    // Create memory buffers on the device
    d_power = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                           size * sizeof(float), h_power, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating power buffer: %d\n", err);
        return -1;
    }
    
    d_temp_src = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            size * sizeof(float), h_temp, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating temp_src buffer: %d\n", err);
        return -1;
    }
    
    d_temp_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                             size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating temp_dst buffer: %d\n", err);
        return -1;
    }
    
    // Load and build the kernel
    size_t kernel_size;
    const char* kernel_source = readKernelFile("hotspot_kernel.cl", &kernel_size);
    
    program = clCreateProgramWithSource(context, 1, &kernel_source, &kernel_size, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating program: %d\n", err);
        return -1;
    }
    
    // Build the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error building program: %s\n", log);
        free(log);
        return -1;
    }
    
    // Create the kernel
    kernel = clCreateKernel(program, "hotspot", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating kernel: %d\n", err);
        return -1;
    }
    
    // Set kernel arguments according to hotspot_kernel.cl signature
    err = clSetKernelArg(kernel, 0, sizeof(int), &iteration);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 0: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_power);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 1: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_temp_src);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 2: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_temp_dst);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 3: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 4, sizeof(int), &grid_cols);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 4: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 5, sizeof(int), &grid_rows);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 5: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 6, sizeof(int), &border_cols);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 6: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 7, sizeof(int), &border_rows);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 7: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 8, sizeof(float), &Cap);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 8: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 9, sizeof(float), &Rx);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 9: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 10, sizeof(float), &Ry);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 10: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 11, sizeof(float), &Rz);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 11: %d\n", err);
        return -1;
    }
    
    err = clSetKernelArg(kernel, 12, sizeof(float), &step);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arg 12: %d\n", err);
        return -1;
    }
    
    // Define work sizes
    size_t local_work_size[2] = {4,4};
    size_t global_work_size[2] = {8,8};
    
    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, 
                                local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error enqueueing kernel: %d\n", err);
        return -1;
    }
    
    // Wait for completion
    clFinish(queue);
    
    // Read back the result from temp_dst (not d_result)
    err = clEnqueueReadBuffer(queue, d_temp_dst, CL_TRUE, 0, size * sizeof(float), 
                             h_result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading back result: %d\n", err);
        return -1;
    }
    
    // Calculate max temperature
    float max_temp = 0.0;
    for (int i = 0; i < size; i++) {
        if (h_result[i] > max_temp) max_temp = h_result[i];
    }
    
    // Print results
    float time_taken = stop_timer(start);
    printf("Hotspot benchmark completed successfully!\n");
    printf("Grid size: %d x %d\n", grid_rows, grid_cols);
    printf("Max temperature: %.2f\n", max_temp);
    printf("Execution time: %.2f ms\n", time_taken);
    
    // Clean up
    clReleaseMemObject(d_temp_src);
    clReleaseMemObject(d_power);
    clReleaseMemObject(d_temp_dst);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(h_temp);
    free(h_power);
    free(h_result);
    free((void*)kernel_source);
    
    return 0;
}
