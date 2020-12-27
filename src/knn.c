
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cblas.h>

#include "knn.h"
#include "utils.h"

#define BLOCK_SIZE 2

knnresult make_knnresult(int m, int k) {
   knnresult ret;

   ret.m = m;
   ret.k = k;
   ret.nidx = (int*) malloc(m*k*sizeof(int));
   ret.ndist = (double*) malloc(m*k*sizeof(double));

   return ret;
}

void free_knnresult(knnresult result) {
    free(result.nidx);
    free(result.ndist);
}

void print_knnresult(knnresult result) {
    printf("m: %d\n", result.m);
    printf("k: %d\n", result.k);
    
    for (int i = 0; i < result.m; i++) {
        printf("\tpoint index: %d\n", i);
        for (int j = 0; j < result.k; j++) {
            printf("\t\tidx: %d, distance: %lf\n", result.nidx[i*result.k+j], result.ndist[i*result.k+j]);
        }
    }
}

void swap_(double* dist, int* idx, int idx1, int idx2) {
    double tmp_dist = dist[idx1];
    int tmp_idx = idx[idx1];
    dist[idx1] = dist[idx2];
    idx[idx1] = idx[idx2];
    dist[idx2] = tmp_dist;
    idx[idx2] = tmp_idx;
}

// hoare partition scheme https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
int partition_(double* dist, int* idx, int lo, int hi) {
    double pivot = dist[(hi+lo)/2];
    int i = lo-1;
    int j = hi;

    while(1) {
        do {
            i++;
        } while (dist[i] < pivot);
        do {
            j--;
        } while (dist[j] > pivot);
        if (i >=j) {
            return j;
        }
        swap_(dist, idx, i, j);
    }
}

// private, recursive part of qselect
void qselect(double* dist, int* idx, int k, int lo, int hi) {
    if (lo < hi) {
        int p = partition_(dist, idx, lo, hi);
        if (p > k)
            qselect(dist, idx, k, lo, p-1);
        else {
            qselect(dist, idx, k, p+1, hi);
        }
    }
    return;
}


// merges two knnresults in-place
void merge_knnresults(knnresult* to, knnresult from) {
    assert(to->k == from.k);

    int m_total = to->m+from.m;
    to->nidx = realloc(to->nidx, m_total*to->k*sizeof(int));
    to->ndist = realloc(to->ndist, m_total*to->k*sizeof(double));
    memcpy(to->nidx+to->m*to->k, from.nidx, from.m*from.k*sizeof(int));
    memcpy(to->ndist+to->m*to->k, from.ndist, from.m*from.k*sizeof(double));
    to->m = m_total;
}

// private function that actually computes kNN of a single block
knnresult kNN_(double * X, double * Y, int n, int m, int d, int k) {
    knnresult result = make_knnresult(m,k);
    double* d_mat = dist_mat(X, Y, n, m, d);
    // tmp vars so that d_mat is not modified
    double* dist_cpy = (double*) malloc(n*sizeof(double));
    int* dist_idx = (int*) malloc(n*sizeof(int));

    for (int i = 0; i < m; i++) {
        // Initially, copy distances to result as-is
        for (int j = 0; j < n; j++) {
            dist_cpy[j] = d_mat[j*m+i]; // i-th column in the matrix
            dist_idx[j] = j;
        }
        // qselect
        qselect(dist_cpy, dist_idx, k, 0, n);

        // copy k first elems to result
        memcpy(result.ndist+i*k, dist_cpy, k*sizeof(double));
        memcpy(result.nidx+i*k, dist_idx, k*sizeof(int));
    }

    free(dist_cpy);
    free(dist_idx);
    free(d_mat);

    return result;
}

knnresult kNN(double * X, double * Y, int n, int m, int d, int k) {
    int offset = 0;

    knnresult first_res;
    int blk_size = 0;
    if (offset+2*BLOCK_SIZE >= m) {
        // near end, m-offset block size
        blk_size = m-offset;
    } else {
        blk_size = BLOCK_SIZE;
    }
    first_res = kNN_(X, Y+offset*d, n, blk_size, d, k);
    offset += blk_size;

    while (offset < m) {
        if (offset+2*BLOCK_SIZE >= m) {
            // near end, m-offset block size
            blk_size = m-offset;

        } else {
            blk_size = BLOCK_SIZE;
        }

        knnresult res = kNN_(X, Y+offset*d, n, blk_size, d, k);
        // merge with first result
        merge_knnresults(&first_res, res);
        free_knnresult(res);

        offset += blk_size;
    }
    return first_res;
}

/// computes distance matrix
/// allocates n*m*sizeof(double)
/// caller is responsible for freeing allocated memory
double* dist_mat(double * X, double * Y, int n, int m, int d) {
    double* d_mat = (double*) malloc(n*m*sizeof(double));
    for (int i = 0; i < n*m; i++) d_mat[i] = 0;
    
    double* x_sq_norm = (double*) malloc(n*m*sizeof(double));
    for (int i = 0; i < n*m; i++) x_sq_norm[i] = 0;

    // X hadamard X
    double* x_sq = (double*) malloc(n*d*sizeof(double));
    for (int i = 0; i < n*d; i++) {
        x_sq[i] = X[i]*X[i];
    }

    // e*e^T aka sum-broadcast
    double* e_et = (double*) malloc(d*m*sizeof(double)); 
    for (int i = 0; i < d*m; i++) {
        e_et[i] = 1;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, d, 1, x_sq, d, e_et, m,
                0, x_sq_norm, m);
    free(x_sq);
    free(e_et);

    double* y_sq_norm = (double*) malloc(n*m*sizeof(double));
    for (int i = 0; i < n*m; i++) y_sq_norm[i] = 0;

    // Y hadamard Y transpose
    double* y_sq = (double*) malloc(d*m*sizeof(double));
    for (int i = 0; i < d*m; i++) {
        int row = i/m;
        int col = i%m;
        int i_t = col*d+row;
        y_sq[i] = Y[i_t]*Y[i_t];
    }

    // e^T*e aka broadcast-sum
    double* et_e = (double*) malloc(n*d*sizeof(double)); 
    for (int i = 0; i < n*d; i++) {
        et_e[i] = 1;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, d, 1, et_e, d, y_sq, m,
                0, y_sq_norm, m);
    free(y_sq);
    free(et_e);

    for (int i = 0; i < n*m; i++) {
        d_mat[i] = x_sq_norm[i] + y_sq_norm[i];
    }
    free(x_sq_norm);
    free(y_sq_norm);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d,
                1, d_mat, m);

    return d_mat;
}
