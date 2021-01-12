
#include <stdio.h>

#include "vptree.h"
#include "utils.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("too few arguments %d\n", argc);
    }

    int n, d;

    double* X = read_csv_range(argv[1], ",", &n, &d, 0, 100, 0, 2);
    vptree* vpt = make_vptree(X, n, d, 5);

    validate_vptree(vpt);
    free_vptree(vpt);
}
