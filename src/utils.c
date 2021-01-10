
#include "utils.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

void print_mat(double* X, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", X[i*cols+j]);
        }
        printf("\n");
    }
}

/**
 * Reads given ranges from csv file and tries to return a row-major matrix. 
 *
 * The matrix is not
 * guaranteed to have end_row-start_row and end_col-start_col columns, but stops it it 
 * encounters end of line or end of file. End index is exclusive. Assumes all rows have the same
 * number of elements
 */
double* read_csv_range(char* path, char* delim, int* n, int* d, int start_row, int end_row, int start_col, int end_col) {
    double* X = (double*) malloc((end_row-start_row)*(end_col-start_col)*sizeof(double));
    FILE* stream = fopen(path, "r");
    if (!stream) {
        perror("Failed to open CSV file");
        exit(1);
    }

    char* line = NULL;
    size_t len = 0;
    int row_num = 0;
    int idx = 0;
    while (getline(&line, &len, stream) != -1 && row_num < end_row) {
        if (row_num < start_row) {
            row_num++;
            continue;
        }
        char* tmp = strdup(line);

        const char* tok;
        int col_num = 0;
        char delimnln[5];
        strcpy(delimnln, delim);
        strcat(delimnln, "\n");
        for (tok = strtok(tmp, delim); tok && *tok && col_num < end_col; tok = strtok(NULL, delimnln)) {
            if (col_num < start_col) {
                col_num++;
                continue;
            }
            // add to mat
            X[idx] = strtod(tok, NULL);
            idx++;
            col_num++;
        }

        // change mat size if cols less than end_col-start_col at first read row
        *d = col_num-start_col;
        if (col_num != end_col) {
            printf("warning: matrix columns count %d less than expected %d, adjusting matrix size\n",
                    (end_col-start_col), *d);
            X = realloc(X, (end_row-start_row)*(*d)*sizeof(double));
        } else {
            assert(*d == end_col-start_col);
        }

        free(tmp);
        row_num++;
    }
    // change mat size if rows less than end_row-start_row at first read row
    *n = row_num-start_row;
    if (row_num != end_row) {
        printf("warning: matrix rows count %d less than expected %d, adjusting matrix size\n",
                (end_row-start_row), *n);
        X = realloc(X, (*n)*(*d)*sizeof(double));
    } else {
        assert(*n == end_row-start_row);
    }

    if (line) free(line);
    fclose(stream);
    return X;
}

int line_count(char* path) {
    int lines = 0;
    FILE* fp = fopen(path, "r");
    if (!fp) {
        perror("Failed to open file");
        exit(1);
    }
    char ch;
    while(!feof(fp)) {
        ch = fgetc(fp);
        if(ch == '\n') {
            lines++;
        }
    }
    fclose(fp);
    return lines;
}

int csv_col_count(char* path, char* delim) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        perror("Failed to open CSV file");
        exit(1);
    }
    int col_cnt = 0;
    char delimnln[5];
    strcpy(delimnln, delim);
    strcat(delimnln, "\n");
    char* line = NULL;
    size_t len = 0;
    char* tok;
    getline(&line, &len, fp);
    char* tmp = strdup(line);
    for (tok = strtok(tmp, delim); tok && *tok; tok = strtok(NULL, delimnln)) {
        col_cnt++;
    }

    free(tmp);
    if (line) free(line);
    fclose(fp);
    return col_cnt;
}

double* read_csv(char* path, char* delim, int* n, int* d) {
    return read_csv_range(path, delim, n, d, 0, line_count(path), 0, csv_col_count(path, delim));
}

double* read_corel_image_features(char* path, int* n, int* d) {
    char* delim = " ";
    return read_csv_range(path, delim, n, d, 0, line_count(path), 1, csv_col_count(path, delim));
}

double* read_miniboone(char* path, int* n, int* d) {
    char* delim = " ";
    return read_csv_range(path, delim, n, d, 1, line_count(path), 0, csv_col_count(path, delim));
}

double* read_audio_features(char* path, int* n, int* d) {
    char* delim = ",";
    return read_csv_range(path, delim, n, d, 4, line_count(path), 0, 20); // limit to 20 dimensions
}

// reads first 17 columns only
double* read_commercial_data(char* path, int* n, int* d) {
    int lines = line_count(path);
    *d = 17;
    *n = lines;
    int start_col = 1; // skip first number
    int end_col = start_col + *d;
    double* X = (double*) malloc((*d)*(*n)*sizeof(double));
    FILE* stream = fopen(path, "r");
    if (!stream) {
        perror("Failed to open CSV file");
        exit(1);
    }
    char* delim = " ";

    char* line = NULL;
    size_t len = 0;
    int row_num = 0;
    int idx = 0;
    while (getline(&line, &len, stream) != -1) {
        char* tmp = strdup(line);

        const char* tok;
        int col_num = 0;
        char delimnln[5];
        strcpy(delimnln, delim);
        strcat(delimnln, "\n");
        for (tok = strtok(tmp, delim); tok && *tok && col_num < end_col; tok = strtok(NULL, delimnln)) {
            if (col_num < start_col) {
                col_num++;
                continue;
            }
            // add to mat the number after :
            X[idx] = strtod(strchr(tok, ':')+1, NULL);
            idx++;
            col_num++;
        }

        // change mat size if cols less than end_col-start_col at first read row
        assert(*d == col_num-start_col);

        free(tmp);
        row_num++;
    }
    assert(lines == row_num);

    if (line) free(line);
    fclose(stream);
    return X;
}

double* load_data(char* path, int type, int* n, int* d) {
    if (type == 0) {
        return read_corel_image_features(path, n, d);
    } else if (type == 1) {
        return read_miniboone(path, n, d);
    } else if (type == 2) {
        return read_audio_features(path, n, d);
    } else if (type == 3) {
        return read_commercial_data(path, n, d);
    } else if (type == 4) {
        return read_csv(path, ",", n, d);
    } else {
        printf("Error: Unknown dataset type %d, exiting\n", type);
        exit(1);
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

void qselect(double* dist, int* idx, int k, int lo, int hi) {
    if (lo < hi) {
        int p = partition_(dist, idx, lo, hi);
        if (p == k-1)
            return;
        else if (p > k-1)
            qselect(dist, idx, k, lo, p-1);
        else
            qselect(dist, idx, k, p+1, hi);
    }
    return;
}

double distance(double* A, double* B, int d) {
    double dist = 0;
    for (int i = 0; i < d; i++) {
        dist += (A[i] - B[i]) * (A[i] - B[i]);
    }
    dist = sqrt(dist);
    return dist;
}
