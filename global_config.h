#ifndef GLOBAL_CONFIG
#define GLOBAL_CONFIG

// General note: make sure config is consistent with whatever kernels are used

// pagerank parameters
#define DAMPENING 0.85 // teleportation probability
#define CHECK_CONVERGENCE 0 // if enabled, algorithm will run till converged, otherwise till max_iter
#define EPSILON 0.0000002 // max. difference at convergence (L2 norm)
#define MAX_ITER 200 // max. iterations of algorithm, set to 0 to disable ceiling
// DO NOT DISABLE BOTH CHECK_CONVERGENCE AND MAX_ITER!

// OCL worker allocation parameters
#define WARP_SIZE 16
#define WORKGROUP_SIZE 256

// other parameters
#define MAX_SOURCE_SIZE (16384)
#define PRINT   0

#endif
