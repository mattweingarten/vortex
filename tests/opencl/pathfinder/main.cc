/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <cstring>
//#include "OpenCL.h"
#include "timing.h"
extern "C" void vortex_rtlsim_dump_stats();

using namespace std;

// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
// #define BENCH_PRINT
#define IN_RANGE(x, min, max)	((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

// Program variables.
int   rows = -1, cols = -1;
int   Ne = rows * cols;
int*  h_data;
int** wall;
int*  result;
int   pyramid_height = -1;
int   verbose = 0;

// OCL config
int platform_id_inuse = 0;            // platform id in use (default: 0)
int device_id_inuse = 0;              //device id in use (default : 0)

#ifdef TIMING
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
	struct timeval tv_init_end;
	struct timeval tv_h2d_start, tv_h2d_end;
	struct timeval tv_d2h_start, tv_d2h_end;
	struct timeval tv_kernel_start, tv_kernel_end;
	struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
	struct timeval tv_close_start, tv_close_end;
	float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
		  d2h_time = 0, close_time = 0, total_time = 0;
#endif

void init(int argc, char** argv)
{
    int cur_arg;
	for (cur_arg = 1; cur_arg<argc; cur_arg++) {
        if (strcmp(argv[cur_arg], "-c") == 0) {
            if (argc >= cur_arg + 1) {
                cols = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-r") == 0) {
            if (argc >= cur_arg + 1) {
                rows = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-h") == 0) {
            if (argc >= cur_arg + 1) {
                pyramid_height = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        if (strcmp(argv[cur_arg], "-v") == 0) {
			verbose = 1;
        }
        else if (strcmp(argv[cur_arg], "-p") == 0) {
            if (argc >= cur_arg + 1) {
                platform_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-d") == 0) {
            if (argc >= cur_arg + 1) {
                device_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
    }

	if (cols < 0 || rows < 0 || pyramid_height < 0)
	{
        fprintf(stderr, "usage: %s <-r rows> <-c cols> <-h pyramid_height> [-v] [-p platform_id] [-d device_id] [-t device_type]\n", argv[0]);
		exit(0);
	}
	h_data = new int[rows * cols];
	wall = new int*[rows];
	for (int n = 0; n < rows; n++)
	{
		// wall[n] is set to be the nth row of the data array.
		wall[n] = h_data + cols * n;
	}
	result = new int[cols];

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			wall[i][j] = rand() % 10;
		}
	}
#ifdef BENCH_PRINT
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%d ", wall[i][j]);
		}
		printf("\n");
	}
#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
}

int main(int argc, char** argv)
{
	init(argc, argv);
	
	// Pyramid parameters.
	int borderCols = (pyramid_height) * HALO;
	// int smallBlockCol = ?????? - (pyramid_height) * HALO * 2;
	// int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

	
	/* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
	   pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

	int size = rows * cols;

	// Create and initialize the OpenCL object.
	//OpenCL cl(verbose);  // 1 means to display output (debugging mode).

#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif
	//cl.init();    // 1 means to use GPU. 0 means use CPU.

	//cl.gwSize(rows * cols);

	// Create and build the kernel.
	string kn = "dynproc_kernel";  // the kernel name, for future use.
	//cl.createKernel(kn);
	///////////////////////////////////////////
	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;

	// Initialize OpenCL
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to find a platform!\n");
		return EXIT_FAILURE;
	}
cl_device_type device_type = CL_DEVICE_TYPE_ALL;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	if (!context) {
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	commands = clCreateCommandQueue(context, device, 0, &err);
	if (!commands) {
		printf("Error: Failed to create a command queue!\n");
		return EXIT_FAILURE;
	}

	//////////////////////////////////////////

	FILE *kernel_fp = fopen("kernel.cl", "r");
	fseek(kernel_fp, 0, SEEK_END);
	size_t kernel_size = ftell(kernel_fp);
	rewind(kernel_fp);

	char *kernel_src = (char*)malloc(kernel_size + 1);
	kernel_src[kernel_size] = '\0';
	fread(kernel_src, sizeof(char), kernel_size, kernel_fp);
	fclose(kernel_fp);

	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_src, NULL, &err);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "dynproc_kernel", &err);

	///////////////////////////////////////////
#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_total_start, &tv);
	init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cl_int* h_outputBuffer = (cl_int*)malloc(16384 * sizeof(cl_int));
    for (int i = 0; i < 16384; i++) {
        h_outputBuffer[i] = 0;
	}

#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_start, NULL);
#endif
	// Allocate device memory.
    cl_mem d_gpuWall = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(cl_int) * (size - cols), NULL, NULL);

    cl_mem d_gpuResult[2];

    d_gpuResult[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(cl_int) * cols, NULL, NULL);

    d_gpuResult[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(cl_int) * cols, NULL, NULL);

    cl_mem d_outputBuffer = clCreateBuffer(context,
        CL_MEM_READ_WRITE, sizeof(cl_int) * 16384, NULL, NULL);

#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_mem_alloc_start, &tv);
    mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cl_event write_event[3];
    clEnqueueWriteBuffer(commands, d_gpuWall, 1, 0,
        sizeof(cl_int) * (size - cols), (h_data + cols), 0, 0, &write_event[0]);

    clEnqueueWriteBuffer(commands, d_gpuResult[0], 1, 0,
        sizeof(cl_int) * cols, h_data, 0, 0, &write_event[1]);

    clEnqueueWriteBuffer(commands, d_outputBuffer, 1, 0,
        sizeof(cl_int) * 16384, h_outputBuffer, 0, 0, &write_event[2]);

#ifdef TIMING
    h2d_time += probe_event_time(write_event[0], commands);
    h2d_time += probe_event_time(write_event[1], commands);
    h2d_time += probe_event_time(write_event[2], commands);
#endif

	int src = 1, final_ret = 0;
	for (int t = 0; t < rows - 1; t += pyramid_height)
	{
		int temp = src;
		src = final_ret;
		final_ret = temp;

		// Calculate this for the kernel argument...
		int arg0 = MIN(pyramid_height, rows-t-1);
		int theHalo = HALO;
		size_t local_size = 256; // or another suitable value

		// Set the kernel arguments.
		clSetKernelArg(kernel, 0,  sizeof(cl_int), (void*) &arg0);
		clSetKernelArg(kernel, 1,  sizeof(cl_mem), (void*) &d_gpuWall);
		clSetKernelArg(kernel, 2,  sizeof(cl_mem), (void*) &d_gpuResult[src]);
		clSetKernelArg(kernel, 3,  sizeof(cl_mem), (void*) &d_gpuResult[final_ret]);
		clSetKernelArg(kernel, 4,  sizeof(cl_int), (void*) &cols);
		clSetKernelArg(kernel, 5,  sizeof(cl_int), (void*) &rows);
		clSetKernelArg(kernel, 6,  sizeof(cl_int), (void*) &t);
		clSetKernelArg(kernel, 7,  sizeof(cl_int), (void*) &borderCols);
		clSetKernelArg(kernel, 8,  sizeof(cl_int), (void*) &theHalo);
		clSetKernelArg(kernel, 9,  sizeof(cl_int) * local_size, 0);
		clSetKernelArg(kernel, 10, sizeof(cl_int) * local_size, 0);
		clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*) &d_outputBuffer);
		//cl.launch(kn);
		size_t global_size = cols;
		
		err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
		clFinish(commands);
		printf("result[0]=%d, result[%d]=%d\n", result[0], cols-1, result[cols-1]);


	}

	// Copy results back to host.
	cl_event event;
	clEnqueueReadBuffer(commands,                   // The command queue.
	                    d_gpuResult[final_ret],   // The result on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_int)*cols,      // Size to copy.
	                    result,                   // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    &event);                  // Event object for determining status. Not used.
#ifdef TIMING
    d2h_time += probe_event_time(event,commands);
#endif
    clReleaseEvent(event);

	// Copy string buffer used for debugging from device to host.
	clEnqueueReadBuffer(commands,                   // The command queue.
	                    d_outputBuffer,           // Debug buffer on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_char)*16384,    // Size to copy.
	                    h_outputBuffer,           // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    &event);                  // Event object for determining status. Not used.
#ifdef TIMING
    d2h_time += probe_event_time(event,commands);
#endif

	// Tack a null terminator at the end of the string.
	h_outputBuffer[16383] = '\0';
	
#ifdef BENCH_PRINT
	for (int i = 0; i < cols; i++)
		printf("%d ", h_data[i]);
	printf("\n");
	for (int i = 0; i < cols; i++)
		printf("%d ", result[i]);
	printf("\n");
#endif

#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif

	clReleaseMemObject(d_gpuWall);
	clReleaseMemObject(d_gpuResult[0]);
	clReleaseMemObject(d_gpuResult[1]);
	clReleaseMemObject(d_outputBuffer);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	vortex_rtlsim_dump_stats();

#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	// Memory cleanup here.
	delete[] h_data;
	delete[] wall;
	delete[] result;

	return EXIT_SUCCESS;
}
