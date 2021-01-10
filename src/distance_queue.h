
#ifndef DISTANCE_QUEUE_H
#define DISTANCE_QUEUE_H

/**
 * Distance queue implementation with integer elements. Pretty much similar to a priority queue
 * but smaller distances have higher priority. The only added feature is that it always
 * keeps \param capacity number of elements. If an element is added when 
 * the queue is full, the item with the lowest priority (highest distance) is discarded
*/
typedef struct distance_queue {
    // i-th elem corresponds to i-th dist. 
    int* elems;
    double* dist;
    // current number of elements
    int size;
    // maximum number of elements
    int capacity;
} distance_queue;

distance_queue* make_distance_queue(int capacity);
void free_distance_queue(distance_queue* queue);
/**
 * pushes \param elem with distance \param dist into the \param q (O(logn))
 */
void dqueue_push(distance_queue* q, int elem, double dist);
/**
 * get lowest distance element (O(logn))
 * \param   q    (input)    the queue of interest
 * \param   elem (output)   element of lowest distance
 * \param   dist (output)   lowest distance
 */
void top(distance_queue* q, int* elem, double* dist);

#endif
