
#include "utils.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

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
double* read_csv_range(char* path, int* n, int* d, int start_row, int end_row, int start_col, int end_col) {
    double* X = (double*) malloc((end_row-start_row)*(end_col-start_col)*sizeof(double));
    FILE* stream = fopen(path, "r");

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
        for (tok = strtok(tmp, ","); tok && *tok && col_num < end_col; tok = strtok(NULL, ",\n")) {
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
            realloc(X, (end_row-start_row)*(*d)*sizeof(double));
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
        realloc(X, (*n)*(*d)*sizeof(double));
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

int csv_col_count(char* path) {
    FILE* fp = fopen(path, "r");
    int col_cnt = 0;
    char* line = NULL;
    size_t len = 0;
    char* tok;
    getline(&line, &len, fp);
    char* tmp = strdup(line);
    for (tok = strtok(tmp, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        col_cnt++;
    }

    free(tmp);
    if (line) free(line);
    fclose(fp);
    return col_cnt;
}

double* read_csv(char* path, int* n, int* d) {
    return read_csv_range(path, n, d, 0, line_count(path), 0, csv_col_count(path));
}
