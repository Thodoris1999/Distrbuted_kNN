
#include <stdio.h>

#include "utils.h"
#include "knn.h"

int main(int argc, char** argv) {
    int n,d;

    if (argc < 2) {
        printf("too few arguments %d\n", argc);
    }

    double* X = read_csv_range(argv[1], &n, &d, 0, 5, 0, 2);
    print_mat(X, n, d);

    knnresult knnres = kNN(X, X, n, n, d, 3);
    print_knnresult(knnres);
}
