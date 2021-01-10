
#include <stdlib.h>
#include <assert.h>

#include "utils.h"

#include "distance_queue.h"

distance_queue* make_distance_queue(int capacity) {
    distance_queue* q = (distance_queue*) malloc(sizeof(distance_queue));
    q->capacity = capacity;
    q->size = 0;
    q->elems = (int*) malloc(capacity*sizeof(int));
    q->dist = (double*) malloc(capacity*sizeof(double));
    return q;
}

void free_distance_queue(distance_queue* queue) {
    free(queue->elems);
    free(queue->dist);
    free(queue);
}

void dqueue_push(distance_queue* q, int elem, double dist) {
    if (q->size < q->capacity) {
        q->elems[q->size] = elem;
        q->dist[q->size] = dist;
        q->size++;
    } else {
        // quick-select to place highest distance at the end of the array
        qselect(q->dist, q->elems, q->size-1, 0, q->size);
        if (dist < q->dist[q->size-1]) {
            q->elems[q->size-1] = elem;
            q->dist[q->size-1] = dist;
        }
    }
}

void top(distance_queue* q, int* elem, double* dist) {
    assert(q->size > 0);

    if (q->size > 1)
        qselect(q->dist, q->elems, 1, 0, q->size);

    *elem = q->elems[0];
    *dist = q->dist[0];
}
