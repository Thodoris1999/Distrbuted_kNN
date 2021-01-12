
#ifndef VPTREE_H
#define VPTREE_H

#define MAX_B 10

#include "knn.h"
#include "distance_queue.h"

typedef struct vptree {
    struct vptree* inner;
    struct vptree* outer;
    // point dimension
    int d;
    // num elements if leaf, -1 otherwise
    int b;
    // leaf element indices in the original array (NULL if not leaf) [b]
    int* idx;
    // leaf element coordinates (NULL if not leaf) [b-by-d]
    double* B;
    // vantage point index
    int vpidx;
    // vantage point coordinates
    double* vpcoo;
    // median
    double m;

} vptree;

// creates VPT. Can also take offset \param offx if X is part of a bigger array
vptree* make_vptree(double* X, int n, int d, int b_max);
void free_vptree(vptree* tree);

distance_queue* vptree_search_knn(vptree* tree, double* querycoo, int k, int offx);
knnresult vptree_search_knn_many(vptree* tree, double* Y, int m, int k, int offx);
void validate_vptree(vptree* node);

#endif
