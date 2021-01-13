
#ifndef KNN_H
#define KNN_H

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

void print_knnresult(knnresult result);

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  \param  X      Corpus data points              [n-by-d]
  \param  Y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);
knnresult kNN_brute_force(double * X, double * Y, int n, int m, int d, int k);
//! k-NN which also allows takes into account that X may be part of a larger array
/*!
  \param offx   offset of corpus points          [scalar]
*/
knnresult kNN_off(double * X, double * Y, int n, int m, int d, int k, int offx);
double* dist_mat(double * X, double * Y, int n, int m, int d);

// merge two knnresults in-place. \param to is changed to contain the k-NN from both \param to and
// \param from
void merge_knnresults(knnresult* to, knnresult from);

knnresult make_knnresult(int m, int k);
void free_knnresult(knnresult knn_res);

#endif
