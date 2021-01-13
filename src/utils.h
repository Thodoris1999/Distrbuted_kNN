
#ifndef UTILS_H
#define UTILS_H

void print_mat(double* X, int rows, int cols);
void print_arr(double* X, int n);

/**
 * Reads given ranges from csv file and tries to return a row-major matrix. 
 *
 * The matrix is not
 * guaranteed to have end_row-start_row and end_col-start_col columns, but stops it it 
 * encounters end of line or end of file. End index is exclusive. Assumes all rows have the same
 * number of elements
 */
double* read_csv_range(char* path, char* delim, int* n, int* d, int start_row, int end_row, int start_col, int end_col);

int line_count(char* path);

int csv_col_count(char* path, char* delim);

double* read_csv(char* path, char* delim, int* n, int* d);
double* read_corel_image_features(char* path, int* n, int* d);
double* read_miniboone(char* path, int* n, int* d, int cols);
double* read_audio_features(char* path, int* n, int* d);
double* read_commercial_data(char* path, int* n, int* d);
double* load_data(char* path, int type, int* n, int* d);

//! Quick select
void qselect(double* dist, int* idx, int k, int lo, int hi);

double distance_sq(double* A, double* B, int d);

#endif
