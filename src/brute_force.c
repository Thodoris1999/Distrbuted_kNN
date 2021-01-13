
#include <stdio.h>
#include <time.h>

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
    
    // file for writing execution time, if provided with 4th argument (time file)
    FILE* fp;
    if (argc == 5) {
        fp = fopen(argv[4], "a");
    }
    // start timer
    struct timespec ts_start, duration;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    knnresult knnres = kNN(X, X, n, n, d, k);

    // end timer
    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    duration.tv_sec = ts_end.tv_sec - ts_start.tv_sec;
    duration.tv_nsec = ts_end.tv_nsec - ts_start.tv_nsec;
    while (duration.tv_nsec > 1000000000) {
        duration.tv_sec++;
        duration.tv_nsec -= 1000000000;
    }
    while (duration.tv_nsec < 0) {
        duration.tv_sec--;
        duration.tv_nsec += 1000000000;
    }
    double dur_d = duration.tv_sec + duration.tv_nsec/1000000000.0;
    printf("%lf", dur_d);
    if (argc == 5 && fp) {
        fprintf(fp, "%lf\n", dur_d);
        fclose(fp);
    }

    print_knnresult(knnres);
    free_knnresult(knnres);
}
