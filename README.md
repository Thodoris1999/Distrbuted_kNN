# Distrbuted_kNN
## Building
`make all`

## Usage
1. For V0: `./bin/v0 <data_file> <type> <k_for_k-NN> <timing_out_file>`
2. For V1: `mpirun -np <nproc> ./bin/v1 <data_file> <type> <k_for_k-NN> <timing_out_file>`
3. For V2: `mpirun -np <nproc> ./bin/v2 <data_file> <type> <k_for_k-NN> <timing_out_file>`

`<timing_out_file>` is optional, it appends a line with the runtime to the specified file.

`<type>` can be:
- 0, for CorelImageFeatures `<datafile>`
- 1, for MiniBoone `<datafile>`
- 2, for audio features "feature.csv" file
- 3, for commercial detection `datafile`
- 4, for a "generic" dataset (assume all are valid numbers, separated with commas and newlines)
