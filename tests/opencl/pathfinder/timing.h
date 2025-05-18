#ifndef TIMING_H
#define TIMING_H

#include <sys/time.h>
#include <CL/opencl.h> // Add this line to include OpenCL headers

inline void tvsub(struct timeval *x, struct timeval *y, struct timeval *result) {
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;
    if (result->tv_usec < 0) {
        result->tv_sec--;
        result->tv_usec += 1000000;
    }
}

inline float probe_event_time(cl_event event, cl_command_queue command_queue) {
    cl_ulong start_time, end_time;
    clFinish(command_queue);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    return (float)((end_time - start_time) * 1.0e-6f);
}

#endif // TIMING_H
