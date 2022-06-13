#ifndef GLOBAL_CONFIG
#define GLOBAL_CONFIG

// pagerank parameters
#define DAMPENING 0.85 // teleportation probability
#define CHECK_CONVERGENCE 0 // if enabled, algorithm will run till converged, otherwise till max_iter
#define EPSILON 0.0000002 // max. difference at convergence (L2 norm)
#define MAX_ITER 100 // max. iterations of algorithm, set to 0 to disable ceiling
// DO NOT DISABLE BOTH CHECK_CONVERGENCE AND MAX_ITER!

// OCL worker allocation parameters
#define WORKGROUP_SIZE 256
#define WARP_SIZE 16

// other parameters
#define MAX_SOURCE_SIZE (16384)
#define PRINT   0

#endif
