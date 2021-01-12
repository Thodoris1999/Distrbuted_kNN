# Distrbuted_kNN
## Building
`make all`

## Usage
1. For V0: `./bin/v0 <data_file> <type> <k_for_k-NN>`
2. For V1: `mpirun -np <nproc> ./bin/v1 <data_file> <type> <k_for_k-NN>`
3. For V2: `mpirun -np <nproc> ./bin/v2 <data_file> <type> <k_for_k-NN>`

Where `<type>` can be:
- 0, for CorelImageFeatures `<datafile>`
- 1, for MiniBoone `<datafile>`
- 2, for audio features "feature.csv" file
- 3, for commercial detection `datafile`
- 4, for a "generic" dataset (assume all are valid numbers, separated with commas and newlines)
