CC=gcc
MPICC=mpicc

BINS_DIR=bin
SOURCES = src/knn.c src/utils.c
V2_SOURCES = src/knn.c src/utils.c src/distance_queue.c src/vptree.c
TEST_VPT_SOURCES = src/knn.c src/utils.c src/distance_queue.c src/vptree.c

CFLAGS=-Wall -O3
CFLAGS_DEBUG=-Wall -g -fsanitize=address
LDFLAGS=-lopenblas -lm

default: all

.PHONY: clean

bin:
	mkdir -p $@

v0: | bin
	$(CC) $(CFLAGS) -o $(BINS_DIR)/$@ $(SOURCES) src/v0.c $(LDFLAGS)

v1: | bin
	$(MPICC) $(CFLAGS) -o $(BINS_DIR)/$@ $(SOURCES) src/v1.c $(LDFLAGS)
        
v2: | bin
	$(MPICC) $(CFLAGS) -o $(BINS_DIR)/$@ $(V2_SOURCES) src/v2.c $(LDFLAGS)

test_vpt: | bin
	$(CC) $(CFLAGS_DEBUG) -o $(BINS_DIR)/$@ $(TEST_VPT_SOURCES) src/test_vptree.c $(LDFLAGS)

all: v0 v1 v2 test_vpt

clean:
	rm -rf $(BINS_DIR)
