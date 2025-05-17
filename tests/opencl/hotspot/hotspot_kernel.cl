#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define BLOCK_SIZE 4 

__kernel void hotspot(int iteration,           // number of iteration
                     global float *power,      // power input
                     global float *temp_src,   // temperature input
                     global float *temp_dst,   // temperature output
                     int grid_cols,            // Col of grid
                     int grid_rows,            // Row of grid
                     int border_cols,          // border offset 
                     int border_rows,          // border offset
                     float Cap,                // Capacitance
                     float Rx, 
                     float Ry, 
                     float Rz, 
                     float step) {
    
    // Get thread IDs
    int tx = get_global_id(0);
    int ty = get_global_id(1);
    
    // Only process if within grid bounds
    if (tx < grid_cols && ty < grid_rows) {
        int index = ty * grid_cols + tx;
        
        float center = temp_src[index];
        float power_val = power[index];
        float result = center;
        
        // Simple stencil calculation for interior points
        if (tx > 0 && tx < grid_cols-1 && ty > 0 && ty < grid_rows-1) {
            float left = temp_src[index - 1];
            float right = temp_src[index + 1];
            float top = temp_src[index - grid_cols];
            float bottom = temp_src[index + grid_cols];
            
            float step_div_Cap = step/Cap;
            float Rx_1 = 1/Rx;
            float Ry_1 = 1/Ry;
            float Rz_1 = 1/Rz;
            float amb_temp = 80.0f;
            
            // Main computation
            result = center + step_div_Cap * (power_val + 
                    (top + bottom - 2.0f * center) * Ry_1 + 
                    (right + left - 2.0f * center) * Rx_1 + 
                    (amb_temp - center) * Rz_1);
        }
        
        // Write result
        temp_dst[index] = result;
    }
}
