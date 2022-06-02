#define WARP_SIZE 16

// computes product between matrix in CSR format and vector using multiple threads per row
// WARP_SIZE must be set correctly to match dynamic worker allocation
__kernel void mCSRmulth(__global const int *rowptr, __global const int *col, __global const float *data,
					    __global float *vin, __global float *vout, __local float *buffer, int rows) {		
	
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
