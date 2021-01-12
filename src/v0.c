
#include <stdio.h>

#include "utils.h"
#include "knn.h"

int main(int argc, char** argv) {
    int n,d;

    if (argc < 4) {
        printf("too few arguments %d\n", argc);
    }

    // double* X = read_csv_range(argv[1], ",", &n, &d, 0, 5, 0, 2);
    double* X = load_data(argv[1], atoi(argv[2]), &n, &d);
    if (n < 1000)
        print_mat(X, n, d);

    int k = atoi(argv[3]);

    knnresult knnres = kNN(X, X, n, n, d, k);
    print_knnresult(knnres);
}
