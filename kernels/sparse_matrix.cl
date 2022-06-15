#define DAMPENING 0.85
#define WARP_SIZE 16

/*
 * GENERAL HELPERS
 */

__kernel void fixPROutput(__global float *vout, int total_nodes) {
	int gid = get_global_id(0);
	int w_total = get_global_size(0);

    int nodeID = gid;
	float adjustment = (1 - DAMPENING) / ((float) total_nodes);
	while (nodeID < total_nodes) {
		vout[nodeID] = DAMPENING * vout[nodeID] + adjustment;
        nodeID += w_total;
    }
}

// WARNING: computes square of norm (does not root the result)
__kernel void normDiff(__global const float *a, __global const float *b, int vec_size, __local float *partial, __global float *res) {
	int gid = get_global_id(0);
	int w_total = get_global_size(0);
	int lid = get_local_id(0);
	int wg_size = get_local_size(0);

    int ind = gid;
	float local_sum = 0.0f;
    float diff;
	while (ind < vec_size) {
        diff = a[ind] - b[ind];
		local_sum += diff*diff;
        ind += w_total;
    }

	partial[lid] = local_sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	int idxStep = 1;
	while(idxStep < wg_size) {
		if(lid % (2*idxStep) == 0)
			partial[lid] += partial[lid+idxStep];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		idxStep *= 2;				
	}

	if(lid == 0)
		atomic_xchg(res, *res + partial[lid]);
		
}


/*
 * MATRIX-VECTOR IMPL.
 */

__kernel void mCSRbasic(__global const int *rowptr, __global const int *col, __global const float *data,
					   	__global const float *vin, __global float *vout, int rows) {		
    
	int gid = get_global_id(0); 
	if(gid < rows) {
		float sum = 0.0f;
        for (int j = rowptr[gid]; j < rowptr[gid + 1]; j++)
            sum += data[j] * vin[col[j]];
		vout[gid] = sum;
	}
}

// computes product between matrix in CSR format and vector using multiple threads per row
// WARP_SIZE must be set correctly to match dynamic worker allocation (same as in global_config.h)
__kernel void mCSRmulth(__global const int *rowptr, __global const int *col, __global const float *data,
					    __global const float *vin, __global float *vout, __local float *buffer, int rows) {		
	
	int lid = get_local_id(0);
	int gid = get_global_id(0);
	int wid = gid / WARP_SIZE;  // warp id
	int wlid = gid % WARP_SIZE; // local id within a warp

	if (wid < rows) {
		buffer[lid] = 0;
		for (int j = rowptr[wid] + wlid; j < rowptr[wid + 1]; j += WARP_SIZE)
			buffer[lid] += data[j] * vin[col[j]];
		barrier(CLK_LOCAL_MEM_FENCE);

		if (wlid < WARP_SIZE/2) {
			for (int inc = WARP_SIZE/2; inc > 0; inc /= 2) {
				buffer[lid] += buffer[lid + inc];
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}

		if (wlid == 0) 
			vout[wid] = buffer[lid];
	}

}

__kernel void mELL(__global const int *col, __global const float *data,
					    __global float *vin, __global float *vout, int rows, int elemsinrow) {		
    
	int gid = get_global_id(0); 
	if(gid < rows) {
		float sum = 0.0f;
		int idx;
		for (int j = 0; j < elemsinrow; j++) {
			idx = j * rows + gid;
            sum += data[idx] * vin[col[idx]];
		}
		vout[gid] = sum;
	}
}
