
#include "vptree.h"

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <float.h>
#include <stdio.h>

#include "utils.h"

// each iteration is responsible of freeing S_idx and allocating S_idx for their children if not leaf
void recurse_make_vp_tree(vptree* node, double* X, int* S_idx, int ns, int d, int b_max) {
    node->d = d;
    if (ns <= b_max) {
        // leaf
        node->inner = NULL;
        node->outer = NULL;
        node->vpcoo = NULL;
        node->b = ns;
        node->idx = (int*) malloc(node->b * sizeof(int));
        memcpy(node->idx, S_idx, node->b * sizeof(int));
        node->B = (double*) malloc(node->b * node->d * sizeof(double));
        for (int i = 0; i < node->b; i++) {
            memcpy(node->B + node->d*i, X+d*node->idx[i], node->d * sizeof(double));
        }
        free(S_idx);
        return;
    }

    node->b = -1; // non-leaf
    node->idx = NULL;
    node->B = NULL;
    // select last point as vantage point for now
    node->vpidx = S_idx[ns-1];
    node->vpcoo = (double*) malloc(d*sizeof(double));
    memcpy(node->vpcoo, X+d*S_idx[ns-1], d*sizeof(double));

    //distances from vantage point
    double* distances = (double*) malloc((ns-1)*sizeof(double));
    for (int i = 0; i < ns-1; i++) {
        distances[i] = distance(X+d*S_idx[i], X+d*node->vpidx, d);
    }

    // partition between median
    int med = (ns-1) / 2;
    qselect(distances, S_idx, med+1, 0, ns-1);

    node->m = distances[med];
    vptree* inner = (vptree*) malloc(sizeof(vptree));
    vptree* outer = (vptree*) malloc(sizeof(vptree));
    node->inner = inner;
    node->outer = outer;
    int* S_idx_inner = (int*) malloc((med-1)*sizeof(int));
    int* S_idx_outer = (int*) malloc((ns-med)*sizeof(int));
    memcpy(S_idx_inner, S_idx, (med-1)*sizeof(int));
    memcpy(S_idx_outer, S_idx+med-1, (ns-med)*sizeof(int));

    free(distances);
    free(S_idx);

    recurse_make_vp_tree(inner, X, S_idx_inner, med-1, d, b_max);
    recurse_make_vp_tree(outer, X, S_idx_outer, ns-med, d, b_max);
}

vptree* make_vptree(double* X, int n, int d, int b_max) {
    vptree* tree = (vptree*) malloc(sizeof(vptree));
    tree->d = d;
    // setup X indices, initially for the whole X
    int* S_idx = (int*) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++) S_idx[i] = i;

    // recursive tree setup
    recurse_make_vp_tree(tree, X, S_idx, n, d, b_max);

    return tree;
}

void recurse_search_vptree(vptree* node, double* querycoo, distance_queue* dqueue, double* tau) {
    int d = node->d;
    if (node->b >= 0) {
        // leaf node. Brute force knn with dqueue + leaf elements as corpus
        if (node->b == 0) return;
        int k = dqueue->capacity;
        double* distances = (double*) malloc((node->b+dqueue->size)*sizeof(double));
        int* indices = (int*) malloc((node->b+dqueue->size)*sizeof(int));
        for (int i = 0; i < node->b; i++) {
            distances[i] = distance(querycoo, node->B + node->d*i, d);
        }
        memcpy(distances+node->b, dqueue->dist, dqueue->size*sizeof(double));
        memcpy(indices, node->idx, node->b*sizeof(int));
        memcpy(indices+node->b, dqueue->elems, dqueue->size*sizeof(int));

        qselect(distances, indices, k, 0, node->b+dqueue->size);
        indices = (int*) realloc(indices, k*sizeof(int));
        distances = (double*) realloc(distances, k*sizeof(double));

        // reconstruct dqueue
        free(dqueue->dist);
        free(dqueue->elems);
        dqueue->dist = distances;
        dqueue->elems = indices;
        return;
    }

    double vpdist = 0;
    for (int i = 0; i < d; i++) {
        vpdist += (querycoo[i] - node->vpcoo[i]) * (querycoo[i] - node->vpcoo[i]);
    }
    vpdist = sqrt(vpdist);

    if (vpdist < *tau) {
        *tau = vpdist;
        dqueue_push(dqueue, node->vpidx, vpdist);
    } else if (dqueue->size < dqueue->capacity) {
        // distance queue has not yet been filled
        dqueue_push(dqueue, node->vpidx, vpdist);
    }

    if (vpdist < node->m) {
        if (vpdist < node->m - *tau) {
            recurse_search_vptree(node->inner, querycoo, dqueue, tau);
        } 
        if (vpdist < node->m + *tau) {
            recurse_search_vptree(node->outer, querycoo, dqueue, tau);
        }
    } else {
        if (vpdist < node->m + *tau) {
            recurse_search_vptree(node->outer, querycoo, dqueue, tau);
        } 
        if (vpdist < node->m - *tau) {
            recurse_search_vptree(node->inner, querycoo, dqueue, tau);
        }
    }
}

/**
 * The knn indices and distances will be written on kidx and kdist, which must be allocated by
 * the caller
 */
distance_queue* vptree_search_knn(vptree* tree, double* querycoo, int k, int offx) {
    distance_queue* dqueue = make_distance_queue(k);
    double tau = DBL_MAX; // set maximum search threshold

    recurse_search_vptree(tree, querycoo, dqueue, &tau);

    // apply offsets
    for (int i = 0; i < dqueue->size; i++) dqueue->elems[i] += offx;
    return dqueue;
}

void validate_vptree(vptree* node) {
    if (node->vpcoo) {
        if (node->inner) {
            if (node->inner->vpcoo) {
                // inner child's vantage point must be closer than median
                double dist = distance(node->vpcoo, node->inner->vpcoo, node->d);
                if (dist >= node->m)
                    printf("Invalid inner child vantage point, distance %lf, median %lf\n", dist, node->m);
            }
            if (node->inner->b >= 0) {
                // leaf inner node. All leaf buckets must be close than median
                for (int i = 0; i < node->inner->b; i++) {
                    double dist = distance(node->vpcoo, node->inner->B+i*node->d, node->d);
                    if (dist >= node->m)
                        printf("Invalid inner child bucket point at idx %d, distance %lf, median %lf\n", i, dist, node->m);
                }
            }
            validate_vptree(node->inner);
        }
        if (node->outer) {
            if (node->outer->vpcoo) {
                // outer child's vantage point must be further than median
                double dist = distance(node->vpcoo, node->outer->vpcoo, node->d);
                if (dist <= node->m)
                    printf("Invalid outer child vantage point, distance %lf, median %lf\n", dist, node->m);
            }
            if (node->outer->b < 0) {
                // leaf inner node. All leaf buckets must be close than median
                for (int i = 0; i < node->outer->b; i++) {
                    double dist = distance(node->vpcoo, node->outer->B+i*node->d, node->d);
                    if (dist <= node->m)
                        printf("Invalid outer child bucket point at idx %d, distance %lf, median %lf\n", i, dist, node->m);
                }
            }
            validate_vptree(node->outer);
        }
    }
}

void free_vptree(vptree* tree) {
    if (tree->vpcoo)
        free(tree->vpcoo);
    if (tree->idx)
        free(tree->idx);
    if (tree->B)
        free(tree->B);
    if (tree->inner) {
        free_vptree(tree->inner);
    }
    if (tree->outer) {
        free_vptree(tree->outer);
    }
    free(tree);
}

void print_vptree(vptree* node) {
    if (node) {
        if (node->b < 0) {
            // non-leaf
            printf("VP at idx %d: ", node->vpidx);
            print_arr(node->vpcoo, node->d);
        } else {
            print_mat(node->B, node->b, node->d);
        }
        print_vptree(node->inner);
        print_vptree(node->outer);
    }
}

knnresult vptree_search_knn_many(vptree* tree, double* Y, int m, int k, int offx) {
    knnresult res = make_knnresult(m, k);
    for (int i = 0; i < m; i++) {
        distance_queue* dqueue = vptree_search_knn(tree, Y+tree->d*i, k, offx);
        memcpy(res.nidx+k*i, dqueue->elems, k*sizeof(int));
        memcpy(res.ndist+k*i, dqueue->dist, k*sizeof(double));
        free_distance_queue(dqueue);
    }
    return res;
}

