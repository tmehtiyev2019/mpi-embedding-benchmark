@echo off
echo Running MPI embedding benchmarks with different process counts...

echo Creating output directories...
mkdir mpi_results\n2
mkdir mpi_results\n4
mkdir mpi_results\n8

echo Running with 2 processes...
mpiexec -n 2 python mpi_embedding_benchmark.py --docs_per_category 512 --min_length 20 --max_length 2000 --batch_sizes 8,16,32,64,128,256,512 --output_dir mpi_results\n2

echo Running with 4 processes...
mpiexec -n 4 python mpi_embedding_benchmark.py --docs_per_category 512 --min_length 20 --max_length 2000 --batch_sizes 8,16,32,64,128,256,512 --output_dir mpi_results\n4

echo Running with 8 processes...
mpiexec -n 8 python mpi_embedding_benchmark.py --docs_per_category 512 --min_length 20 --max_length 2000 --batch_sizes 8,16,32,64,128,256,512 --output_dir mpi_results\n8

echo Generating combined comparison charts...
python -c "from mpi_embedding_benchmark import plot_combined_comparison; plot_combined_comparison('mpi_results', 'combined_comparison')"

echo Done!