
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "utils.h"
#include "knn.h"
#include "vptree.h"

int main(int argc, char** argv) {
    int numnodes, nodeid, next_id, prev_id;
    int d_tag = 1; // dimension data tag
    int n_tag = 2;
    int data_tag = 3;
    int off_tag = 4;
    int k_tag = 4;
    MPI_Status status;

    int nx, ny, nz, d, n, k;
    int offx, offy, offz;
    double* X = NULL;
    double* Y = NULL;
    double* Z = NULL;
    knnresult local_res;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numnodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeid);
    prev_id = nodeid - 1;
    next_id = nodeid + 1;
    if (prev_id < 0)
        prev_id = numnodes - 1;
    if (next_id == numnodes)
        next_id = 0;
    printf("MPI task %d has started...\n", nodeid);

    if (nodeid == 0) {
        // coordinator initialization
        //double* X_all = read_csv_range(argv[1], ",", &n, &d, 0, 25, 0, 4);
        double* X_all = load_data(argv[1], atoi(argv[2]), &n, &d);
        k = atoi(argv[3]);
        assert(n >= k*(numnodes+1)); // required so each process has at least k neighbours
        if (n < 1000)
            print_mat(X_all, n, d);

        int leftovers = n % numnodes;
        int offset = 0;
        MPI_Request** init_data_req = (MPI_Request**) malloc((numnodes-1)*sizeof(MPI_Request*));
        for (int i = 0; i < numnodes-1; i++) init_data_req[i] = (MPI_Request*) malloc(sizeof(MPI_Request));
        for (int i = 0; i < numnodes; i++) {
            // send dimension, total n and k
            if (i != 0) {
                MPI_Send(&d, 1, MPI_INT, next_id, d_tag, MPI_COMM_WORLD);
                MPI_Send(&n, 1, MPI_INT, next_id, n_tag, MPI_COMM_WORLD);
                MPI_Send(&k, 1, MPI_INT, next_id, k_tag, MPI_COMM_WORLD);
            }

            // distribute chunks to all processes
            int chunk_size = n / numnodes;
            if (leftovers > 0) {
                chunk_size++;
                leftovers--;
            }

            if (i != 0) {
                MPI_Send(&offset, 1, MPI_INT, i, off_tag, MPI_COMM_WORLD);
                MPI_Send(&chunk_size, 1, MPI_INT, i, n_tag, MPI_COMM_WORLD);
                MPI_Isend(X_all+offset*d, chunk_size*d, MPI_DOUBLE, i, data_tag, MPI_COMM_WORLD, init_data_req[i-1]);
            } else {
                // no need to send to self, just assign variables
                offx = offset;
                nx = chunk_size;
                X = (double*) malloc(nx*d*sizeof(double));
                memcpy(X, X_all+offx*d, nx*d*sizeof(double));
            }

            offset += chunk_size;
        }
        for (int i = 0; i < numnodes-1; i++) {
            MPI_Wait(init_data_req[i], &status);
            free(init_data_req[i]);
        }
        free(init_data_req);
        free(X_all);
    } else {
        // worker initialization
        // receive dimension, total n and k
        MPI_Recv(&d, 1, MPI_INT, 0, d_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&n, 1, MPI_INT, 0, n_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&k, 1, MPI_INT, 0, k_tag, MPI_COMM_WORLD, &status);

        // receive corpus set chunk from coordinator
        MPI_Recv(&offx, 1, MPI_INT, 0, off_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&nx, 1, MPI_INT, 0, n_tag, MPI_COMM_WORLD, &status);
        X = (double*) malloc(nx*d*sizeof(double));
        MPI_Recv(X, nx*d, MPI_DOUBLE, 0, data_tag, MPI_COMM_WORLD, &status);
    }
    printf("MPI task %d initialized. Doing private work...\n", nodeid);

    // common work
    // build PVT
    vptree* vpt = make_vptree(X, nx, d, 30);
    // set first query set to be the same as first corpus chunk
    offz = offx;
    nz = nx;
    Z = (double*) malloc(nz*d*sizeof(double));
    memcpy(Z, X, nz*d*sizeof(double));

    // setup noblocking request handle
    MPI_Request* off_recv_req = (MPI_Request*) malloc(sizeof(MPI_Request));
    MPI_Request* off_send_req = (MPI_Request*) malloc(sizeof(MPI_Request));
    MPI_Request* n_recv_req = (MPI_Request*) malloc(sizeof(MPI_Request));
    MPI_Request* n_send_req = (MPI_Request*) malloc(sizeof(MPI_Request));
    MPI_Request* data_recv_req = (MPI_Request*) malloc(sizeof(MPI_Request));
    MPI_Request* data_send_req = (MPI_Request*) malloc(sizeof(MPI_Request));

    // k-NN result in which all of the process's results will be combined into
    local_res = make_knnresult(n,k);
    for (int i = 0; i < numnodes; i++) {
        // send corpus chunk to next process
        printf("%d %d start\n", nodeid, i);
        // copy Z to Y
        offy = offz;
        ny = nz;
        Y = (double*) realloc(Y, ny*d*sizeof(double));
        memcpy(Y, Z, ny*d*sizeof(double));

        // send next query set to next process (async)
        if (i != numnodes - 1) { // nothing to send next
            MPI_Isend(&offy, 1, MPI_INT, next_id, off_tag, MPI_COMM_WORLD, off_send_req);
            MPI_Isend(&ny, 1, MPI_INT, next_id, n_tag, MPI_COMM_WORLD, n_send_req);
            MPI_Isend(Y, ny*d, MPI_DOUBLE, next_id, data_tag, MPI_COMM_WORLD, data_send_req);
        }
        printf("%d %d sent data async\n", nodeid, i);

        // request next query set from previous process (async)
        if (i != numnodes - 1) { // nothing to receive otherwise
            MPI_Irecv(&offz, 1, MPI_INT, prev_id, off_tag, MPI_COMM_WORLD, off_recv_req);
            MPI_Irecv(&nz, 1, MPI_INT, prev_id, n_tag, MPI_COMM_WORLD, n_recv_req);
            MPI_Wait(off_recv_req, &status);
            MPI_Wait(n_recv_req, &status);
            Z = (double*) realloc(Z, nz*d*sizeof(double));
            MPI_Irecv(Z, nz*d, MPI_DOUBLE, prev_id, data_tag, MPI_COMM_WORLD, data_recv_req);
        }
        printf("%d %d requested data\n", nodeid, i);

        // perform k-NN
        knnresult tmp_res = vptree_search_knn_many(vpt, Y, ny, k, offx);
        memcpy(local_res.nidx+offy*k, tmp_res.nidx, tmp_res.m*k*sizeof(int));
        memcpy(local_res.ndist+offy*k, tmp_res.ndist, tmp_res.m*k*sizeof(double));
        free_knnresult(tmp_res);
        printf("%d %d knn'd\n", nodeid, i);

        // wait for requested receive
        if (i != numnodes - 1) { // nothing to receive otherwise
            MPI_Wait(data_recv_req, &status);
            MPI_Wait(off_send_req, &status);
            MPI_Wait(n_send_req, &status);
            MPI_Wait(data_send_req, &status);
        }
        printf("%d %d end\n", nodeid, i);
    }
    if (n < 50)
        print_knnresult(local_res);
    free(X);
    free(Z);
    free(off_recv_req);
    free(off_send_req);
    free(n_recv_req);
    free(n_send_req);
    free(data_recv_req);
    free(data_send_req);
    free_vptree(vpt);
    printf("MPI task %d private work done. Sending to coordinator and merging...\n", nodeid);

    if (nodeid == 0) {
        // cordinator finilization

        // get knnresults from all processes and merge
        int** all_idx = (int**) malloc(numnodes*sizeof(int*));
        double** all_dist = (double**) malloc(numnodes*sizeof(double*));
        // MPI request handles
        MPI_Request** fin_idx_req = (MPI_Request**) malloc((numnodes-1)*sizeof(MPI_Request*));
        MPI_Request** fin_dist_req = (MPI_Request**) malloc((numnodes-1)*sizeof(MPI_Request*));
        for (int i = 0; i < numnodes-1; i++) fin_idx_req[i] = (MPI_Request*) malloc(sizeof(MPI_Request));
        for (int i = 0; i < numnodes-1; i++) fin_dist_req[i] = (MPI_Request*) malloc(sizeof(MPI_Request));
        for (int i = 0; i < numnodes; i++) {
            all_idx[i] = (int*) malloc(n*k*sizeof(int));
            all_dist[i] = (double*) malloc(n*k*sizeof(double));

            if (i != 0) {
                // receive from workers
                MPI_Irecv(all_idx[i], n*k, MPI_INT, i, data_tag, MPI_COMM_WORLD, fin_idx_req[i-1]);
                MPI_Irecv(all_dist[i], n*k, MPI_DOUBLE, i, data_tag, MPI_COMM_WORLD, fin_dist_req[i-1]);
            } else {
                // copy from local result
                memcpy(all_idx[i], local_res.nidx, n*k*sizeof(int));
                memcpy(all_dist[i], local_res.ndist, n*k*sizeof(double));
            }
        }
        for (int i = 0; i < numnodes-1; i++) {
            MPI_Wait(fin_idx_req[i], &status);
            free(fin_idx_req[i]);
            MPI_Wait(fin_dist_req[i], &status);
            free(fin_dist_req[i]);
        }
        free(fin_idx_req);
        free(fin_dist_req);

        // merge
        knnresult total_res = make_knnresult(n, k);
        print_knnresult(total_res);
        // idx and dist are tmp vars to hold k-NN from all nodes of an element in X
        int* idx = (int*) malloc(numnodes*k*sizeof(int));
        double* dist = (double*) malloc(numnodes*k*sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < numnodes; j++) {
                memcpy(idx+k*j, all_idx[j]+k*i, k*sizeof(int));
                memcpy(dist+k*j, all_dist[j]+k*i, k*sizeof(double));
            }

            qselect(dist, idx, k, 0, numnodes*k);
            // copy k first elems to final result 
            memcpy(total_res.nidx+k*i, idx, k*sizeof(int));
            memcpy(total_res.ndist+k*i, dist, k*sizeof(double));
        }
        free(idx);
        free(dist);

        for (int i = 0; i < numnodes; i++) {
            free(all_idx[i]);
            free(all_dist[i]);
        }
        free(all_idx);
        free(all_dist);

        if (n < 100)
            print_knnresult(total_res);

        free_knnresult(total_res);
    } else {
        // worker finilization

        // send results back to coordinator
        MPI_Send(local_res.nidx, n*k, MPI_INT, 0, data_tag, MPI_COMM_WORLD);
        MPI_Send(local_res.ndist, n*k, MPI_DOUBLE, 0, data_tag, MPI_COMM_WORLD);
    }
    printf("MPI task %d done. \n", nodeid);
    free_knnresult(local_res);

    MPI_Finalize();
}
