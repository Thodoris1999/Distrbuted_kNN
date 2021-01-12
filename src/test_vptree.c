
#include <stdio.h>
#include <stdlib.h>

#include "vptree.h"
#include "utils.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("too few arguments %d\n", argc);
    }

    int n, d;

    double* X = read_csv_range(argv[1], ",", &n, &d, 0, 50, 0, 2);
    vptree* vpt = make_vptree(X, n, d, 5);

    print_vptree(vpt);

    validate_vptree(vpt);
    free_vptree(vpt);
    free(X);
}
