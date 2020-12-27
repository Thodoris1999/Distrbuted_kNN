CC=gcc

BINS_DIR=bin
SOURCES = src/knn.c src/utils.c

CFLAGS=-Wall -g#-O3
LDFLAGS=-lopenblas

default: all

.PHONY: clean

bin:
	mkdir -p $@

v0: | bin
	$(CC) $(CFLAGS) -o $(BINS_DIR)/$@ $(SOURCES) src/v0.c $(LDFLAGS)

all: v0

clean:
	rm -rf $(BINS_DIR)
