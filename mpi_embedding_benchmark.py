import time
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from mpi4py import MPI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def print_rank0(msg):
    """Print message only from rank 0"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print(msg)


def generate_balanced_test_data(docs_per_category=128, min_length=20, max_length=1000):
    """Generate test documents ensuring a minimum count per size category"""
    # Define size ranges for each category
    small_min, small_max = min_length, min_length + (max_length - min_length) // 5
    medium_min, medium_max = small_max + 1, small_max + (max_length - min_length) // 3
    large_min, large_max = medium_max + 1, max_length
    
    print_rank0(f"Generating balanced document sets:")
    print_rank0(f"  Small: {docs_per_category} docs, length {small_min}-{small_max} words")
    print_rank0(f"  Medium: {docs_per_category} docs, length {medium_min}-{medium_max} words")
    print_rank0(f"  Large: {docs_per_category} docs, length {large_min}-{large_max} words")
    
    # Generate documents for each category
    small_docs = []
    small_sizes = []
    medium_docs = []
    medium_sizes = []
    large_docs = []
    large_sizes = []
    
    # Generate small documents
    for i in range(docs_per_category):
        doc_length = np.random.randint(small_min, small_max + 1)
        doc = ' '.join([f"word{np.random.randint(1000)}" for _ in range(doc_length)])
        small_docs.append(doc)
        small_sizes.append(len(doc))
    
    # Generate medium documents
    for i in range(docs_per_category):
        doc_length = np.random.randint(medium_min, medium_max + 1)
        doc = ' '.join([f"word{np.random.randint(1000)}" for _ in range(doc_length)])
        medium_docs.append(doc)
        medium_sizes.append(len(doc))
    
    # Generate large documents
    for i in range(docs_per_category):
        doc_length = np.random.randint(large_min, large_max + 1)
        doc = ' '.join([f"word{np.random.randint(1000)}" for _ in range(doc_length)])
        large_docs.append(doc)
        large_sizes.append(len(doc))
    
    total_docs = docs_per_category * 3
    print_rank0(f"Generated {total_docs} documents in total")
    
    return {
        'small': (small_docs, small_sizes),
        'medium': (medium_docs, medium_sizes),
        'large': (large_docs, large_sizes)
    }


class MPIEmbeddingPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2', vector_dim=384):
        # Get MPI details
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Start timing model loading
        if self.rank == 0:
            print(f"Rank {self.rank}: Loading embedding model...")
        
        model_start = time.time()
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        model_end = time.time()
        
        if self.rank == 0:
            print(f"Rank {self.rank}: Model loaded in {model_end - model_start:.2f} seconds")
            
        self.vector_dim = vector_dim
        
        if self.rank == 0:
            print(f"Initialized MPIEmbeddingPipeline with {self.size} processes")
    
    def distribute_batch(self, documents, document_sizes):
        """Distribute documents across all processes"""
        distribute_start = time.time()
        
        if self.rank == 0:
            # Calculate workload for each process
            n_docs = len(documents)
            docs_per_process = [n_docs // self.size + (1 if i < n_docs % self.size else 0) 
                               for i in range(self.size)]
            
            # Distribute documents and sizes
            start_idx = 0
            doc_chunks = []
            size_chunks = []
            
            for chunk_size in docs_per_process:
                end_idx = start_idx + chunk_size
                doc_chunks.append(documents[start_idx:end_idx])
                size_chunks.append(document_sizes[start_idx:end_idx])
                start_idx = end_idx
        else:
            doc_chunks = None
            size_chunks = None
        
        # Scatter data to all processes
        scatter_start = time.time()
        my_docs = self.comm.scatter(doc_chunks, root=0)
        my_sizes = self.comm.scatter(size_chunks, root=0)
        scatter_end = time.time()
        
        if self.rank == 0:
            print(f"Rank {self.rank}: Data scatter took {scatter_end - scatter_start:.2f} seconds")
        
        distribute_end = time.time()
        if self.rank == 0:
            print(f"Rank {self.rank}: Total distribution took {distribute_end - distribute_start:.2f} seconds")
            
        return my_docs, my_sizes
    
    def embed_distributed(self, documents, document_sizes):
        """Distribute and embed documents across all processes"""
        start_time = time.time()
        
        # Distribute documents
        distribute_start = time.time()
        my_docs, my_sizes = self.distribute_batch(documents, document_sizes)
        distribute_end = time.time()
        
        # Each process embeds its chunk
        embed_start = time.time()
        if len(my_docs) > 0:
            if self.rank == 0:
                print(f"Rank {self.rank}: Embedding {len(my_docs)} documents (avg size: {np.mean(my_sizes):.1f} chars)")
            my_embeddings = self.model.encode(my_docs)
        else:
            my_embeddings = np.zeros((0, self.vector_dim), dtype=np.float32)
        embed_end = time.time()
        
        if self.rank == 0:
            print(f"Rank {self.rank}: Embedding took {embed_end - embed_start:.2f} seconds")
        
        # Gather all embeddings, documents, and sizes back to rank 0
        gather_start = time.time()
        all_embeddings = self.comm.gather(my_embeddings, root=0)
        all_docs = self.comm.gather(my_docs, root=0)
        all_sizes = self.comm.gather(my_sizes, root=0)
        gather_end = time.time()
        
        if self.rank == 0:
            print(f"Rank {self.rank}: Gathering results took {gather_end - gather_start:.2f} seconds")
        
        # Process gathered data on rank 0
        if self.rank == 0:
            combine_start = time.time()
            # Combine all embeddings
            combined_embeddings = np.vstack(all_embeddings)
            combined_docs = []
            combined_sizes = []
            
            for docs in all_docs:
                combined_docs.extend(docs)
            
            for sizes in all_sizes:
                combined_sizes.extend(sizes)
            combine_end = time.time()
            print(f"Rank {self.rank}: Combining results took {combine_end - combine_start:.2f} seconds")
        
        end_time = time.time()
        if self.rank == 0:
            print(f"Rank {self.rank}: Total distributed embedding took {end_time - start_time:.2f} seconds")
            print(f"  - Distribution: {distribute_end - distribute_start:.2f}s ({((distribute_end - distribute_start)/(end_time - start_time))*100:.1f}%)")
            print(f"  - Embedding: {embed_end - embed_start:.2f}s ({((embed_end - embed_start)/(end_time - start_time))*100:.1f}%)")
            print(f"  - Gathering: {gather_end - gather_start:.2f}s ({((gather_end - gather_start)/(end_time - start_time))*100:.1f}%)")
            print(f"  - Combining: {combine_end - combine_start:.2f}s ({((combine_end - combine_start)/(end_time - start_time))*100:.1f}%)")
            
        return end_time - start_time


def run_sequential_embedding(documents, model_name='all-MiniLM-L6-v2'):
    """Run sequential embedding for baseline comparison"""
    load_start = time.time()
    pipeline = SentenceTransformer(model_name)
    load_end = time.time()
    
    print(f"Sequential: Model loading took {load_end - load_start:.2f} seconds")
    print(f"Sequential: Embedding {len(documents)} documents")
    
    start_time = time.time()
    
    # Process documents in batch
    embeddings = pipeline.encode(documents)
    
    end_time = time.time()
    print(f"Sequential: Embedding completed in {end_time - start_time:.2f} seconds")
    return end_time - start_time


def run_mpi_benchmark(data_categories, batch_sizes):
    """Run benchmark using MPI for different document sizes"""
    # Get MPI details
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    results = {}
    
    # Run benchmark for each document size category
    for category, (docs, sizes) in data_categories.items():
        print_rank0(f"\nRunning MPI benchmark for {category} documents ({len(docs)} samples)")
        
        # Skip if no documents in this category
        if not docs:
            continue
            
        # Create results structure for this category
        if rank == 0:
            category_results = {
                "sequential": {},
                "mpi": {},
                "document_sizes": {
                    "min": min(sizes),
                    "max": max(sizes),
                    "mean": np.mean(sizes),
                    "median": np.median(sizes)
                }
            }
        
        # Filter batch sizes that are larger than document count
        if rank == 0:
            valid_batch_sizes = [b for b in batch_sizes if b <= len(docs)]
            if len(valid_batch_sizes) != len(batch_sizes):
                skipped = [b for b in batch_sizes if b > len(docs)]
                print_rank0(f"  Skipping batch sizes {skipped} as they exceed document count ({len(docs)})")
        else:
            valid_batch_sizes = None
            
        # Broadcast valid batch sizes to all processes
        valid_batch_sizes = comm.bcast(valid_batch_sizes, root=0)
        
        # Sequential processing (only on rank 0)
        if rank == 0:
            print_rank0("Running sequential benchmark as baseline...")
            
            # Measure model loading time outside the loop (do it only once)
            
            # Use the specified batch sizes, not document sizes
            for batch_size in valid_batch_sizes:
                print_rank0(f"  Testing batch size: {batch_size}")
                
                seq_time_total = 0
                
                # Process documents in batches of the specified size
                batch_start = time.time()
                for i in range(0, len(docs), batch_size):
                    end_idx = min(i + batch_size, len(docs))
                    batch = docs[i:end_idx]
                    
                    # Embedding time
                    seq_time = run_sequential_embedding(batch)
                    seq_time_total += seq_time
                    
                batch_end = time.time()
                print_rank0(f"  Sequential processing completed in {batch_end - batch_start:.2f} seconds")
                
                category_results["sequential"][str(batch_size)] = {
                    "total_time": seq_time_total,
                    "docs_per_second": len(docs) / seq_time_total
                }
        
        # MPI processing
        print_rank0("Running MPI benchmark...")
        
        # Use the specified batch sizes, not document sizes
        for batch_size in valid_batch_sizes:
            pipeline = MPIEmbeddingPipeline()
            
            # Barrier to ensure all processes are synchronized
            comm.Barrier()
            
            if rank == 0:
                print_rank0(f"  Testing batch size: {batch_size}")
                mpi_time_total = 0
            
            # Process documents in batches of the specified size
            batch_start = time.time()
            for i in range(0, len(docs), batch_size):
                end_idx = min(i + batch_size, len(docs))
                
                if rank == 0:
                    batch = docs[i:end_idx]
                    batch_sizes_subset = sizes[i:end_idx]
                else:
                    batch = None
                    batch_sizes_subset = None
                
                # Broadcast batch to all processes
                broadcast_start = time.time()
                batch = comm.bcast(batch, root=0)
                batch_sizes_subset = comm.bcast(batch_sizes_subset, root=0)
                broadcast_end = time.time()
                
                if rank == 0:
                    print_rank0(f"    Broadcast took {broadcast_end - broadcast_start:.2f} seconds")
                
                # Add debug prints to track progress
                if rank == 0:
                    print_rank0(f"    Processing batch {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size} with {len(batch)} documents")
                
                # Process batch with MPI
                comm.Barrier()  # Ensure all processes are ready
                elapsed_time = pipeline.embed_distributed(batch, batch_sizes_subset)
                
                if rank == 0:
                    mpi_time_total += elapsed_time
                    print_rank0(f"    Batch completed in {elapsed_time:.2f}s")
                
                # Add small delay to prevent potential race conditions
                time.sleep(0.1)
                comm.Barrier()  # Ensure all processes finish before next batch
            
            if rank == 0:
                batch_end = time.time()
                print_rank0(f"  MPI processing completed in {batch_end - batch_start:.2f} seconds")
            
            # Store results for this batch size
            if rank == 0:
                category_results["mpi"][str(batch_size)] = {
                    "total_time": mpi_time_total,
                    "docs_per_second": len(docs) / mpi_time_total,
                    "speedup": category_results["sequential"][str(batch_size)]["total_time"] / mpi_time_total,
                    "efficiency": (category_results["sequential"][str(batch_size)]["total_time"] / mpi_time_total) / size,
                    "num_processes": size
                }
        
        if rank == 0:
            results[category] = category_results
    
    return results if rank == 0 else None


def plot_mpi_results(results, output_dir="mpi_benchmark_results"):
    """Generate plots showing MPI benchmark results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each document size category
    for category, category_results in results.items():
        # Only use batch sizes that are actually in the results
        batch_sizes = sorted([int(b) for b in category_results["sequential"].keys()])
        
        if not batch_sizes:  # Skip if no results for this category
            continue
        
        print(f"Plotting results for {category} documents with batch sizes: {batch_sizes}")
        
        # Plot 1: Processing time comparison
        plt.figure(figsize=(12, 8))
        x = np.arange(len(batch_sizes))
        width = 0.35
        
        # Plot sequential vs MPI
        plt.bar(x - width/2, [category_results["sequential"][str(b)]["total_time"] for b in batch_sizes], 
               width, label='Sequential')
        plt.bar(x + width/2, [category_results["mpi"][str(b)]["total_time"] for b in batch_sizes], 
               width, label=f'MPI ({category_results["mpi"][str(batch_sizes[0])]["num_processes"]} processes)')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Total Processing Time (s)')
        plt.title(f'Processing Time by Batch Size ({category} documents)')
        plt.xticks(x, batch_sizes)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{category}_processing_time.png'))
        plt.close()
        
        # Plot 2: Documents per second
        plt.figure(figsize=(12, 8))
        
        # Plot sequential vs MPI
        plt.bar(x - width/2, [category_results["sequential"][str(b)]["docs_per_second"] for b in batch_sizes], 
               width, label='Sequential')
        plt.bar(x + width/2, [category_results["mpi"][str(b)]["docs_per_second"] for b in batch_sizes], 
               width, label=f'MPI ({category_results["mpi"][str(batch_sizes[0])]["num_processes"]} processes)')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Documents per Second')
        plt.title(f'Throughput by Batch Size ({category} documents)')
        plt.xticks(x, batch_sizes)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{category}_docs_per_second.png'))
        plt.close()
        
        # Plot 3: Speedup by batch size
        plt.figure(figsize=(12, 8))
        
        speedups = [category_results["mpi"][str(b)]["speedup"] for b in batch_sizes]
        plt.bar(x, speedups, width=0.6)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup (vs Sequential)')
        plt.title(f'MPI Speedup by Batch Size ({category} documents, {category_results["mpi"][str(batch_sizes[0])]["num_processes"]} processes)')
        plt.xticks(x, batch_sizes)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values above bars
        for i, v in enumerate(speedups):
            plt.text(i, v + 0.1, f'{v:.2f}x', ha='center')
        
        plt.savefig(os.path.join(output_dir, f'{category}_speedup.png'))
        plt.close()
        
        # Plot 4: Parallel efficiency
        plt.figure(figsize=(12, 8))
        
        efficiencies = [category_results["mpi"][str(b)]["efficiency"] for b in batch_sizes]
        plt.bar(x, efficiencies, width=0.6)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Parallel Efficiency')
        plt.title(f'MPI Parallel Efficiency by Batch Size ({category} documents, {category_results["mpi"][str(batch_sizes[0])]["num_processes"]} processes)')
        plt.xticks(x, batch_sizes)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values above bars
        for i, v in enumerate(efficiencies):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.savefig(os.path.join(output_dir, f'{category}_efficiency.png'))
        plt.close()
    
    # Create cross-category comparisons
    categories = list(results.keys())
    
    if len(categories) > 1:  # Only if we have multiple categories
        # Find a common batch size across all categories
        common_batch_sizes = set()
        for category in categories:
            if common_batch_sizes:
                common_batch_sizes &= set(results[category]["sequential"].keys())
            else:
                common_batch_sizes = set(results[category]["sequential"].keys())
        
        if common_batch_sizes:
            batch_size = min(int(b) for b in common_batch_sizes)  # Use smallest common batch size
            
            # Plot 5: Document size impact on speedup
            plt.figure(figsize=(12, 8))
            
            # Speedup by document size category
            speedups = [results[cat]["mpi"][str(batch_size)]["speedup"] for cat in categories]
            
            x = np.arange(len(categories))
            plt.bar(x, speedups, width=0.6)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
            
            plt.xlabel('Document Size Category')
            plt.ylabel('Speedup (vs Sequential)')
            plt.title(f'Document Size Impact on MPI Speedup (Batch Size {batch_size})')
            plt.xticks(x, [cat.capitalize() for cat in categories])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values above bars
            for i, v in enumerate(speedups):
                plt.text(i, v + 0.1, f'{v:.2f}x', ha='center')
            
            plt.savefig(os.path.join(output_dir, 'size_impact_speedup.png'))
            plt.close()
    
    print(f"All MPI charts saved to {output_dir} directory")


def plot_combined_comparison(results_dir, output_dir="combined_comparison"):
    """Generate plots comparing sequential, 2-core, 4-core and 8-core performance in the same charts"""
    # Define the process counts we're comparing
    process_counts = [2, 4, 8]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results for each process count
    all_results = {}
    for n_proc in process_counts:
        try:
            with open(os.path.join(results_dir, f'n{n_proc}', 'mpi_benchmark_results.json'), 'r') as f:
                all_results[n_proc] = json.load(f)
                print(f"Loaded results for {n_proc} processes")
        except FileNotFoundError:
            print(f"Warning: No results found for {n_proc} processes")
    
    if not all_results:
        print("No results found for any process count. Exiting.")
        return
    
    # Process each document category
    categories = next(iter(all_results.values())).keys()
    
    for category in categories:
        print(f"Generating combined charts for {category} documents")
        
        # Find common batch sizes across all process counts
        common_batch_sizes = None
        for n_proc, results in all_results.items():
            if category not in results:
                continue
                
            if common_batch_sizes is None:
                common_batch_sizes = set(results[category]["mpi"].keys())
            else:
                common_batch_sizes &= set(results[category]["mpi"].keys())
        
        if not common_batch_sizes:
            print(f"  No common batch sizes found for {category} documents, skipping")
            continue
            
        batch_sizes = sorted(int(b) for b in common_batch_sizes)
        print(f"  Using batch sizes: {batch_sizes}")
        
        # 1. Combined Processing Time Chart
        plt.figure(figsize=(14, 8))
        
        # Create a bar grouping for each batch size
        x = np.arange(len(batch_sizes))
        width = 0.2  # Width of each bar
        
        # Get sequential times as reference
        ref_proc = process_counts[0]  # Use the first process count's sequential data
        if ref_proc in all_results and category in all_results[ref_proc]:
            seq_times = [all_results[ref_proc][category]["sequential"][str(b)]["total_time"] for b in batch_sizes]
            plt.bar(x - 1.5*width, seq_times, width, label=f'Sequential', color='gray')
        
        # Add MPI times for each process count
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, n_proc in enumerate(process_counts):
            if n_proc not in all_results or category not in all_results[n_proc]:
                continue
                
            results = all_results[n_proc][category]
            # Adjust position to create grouped bars
            pos = x - 0.5*width + i*width
            times = [results["mpi"][str(b)]["total_time"] for b in batch_sizes]
            plt.bar(pos, times, width, label=f'{n_proc} processes', color=colors[i])
        
        plt.xlabel('Batch Size', fontsize=14)
        plt.ylabel('Processing Time (s)', fontsize=14)
        plt.title(f'Processing Time Comparison ({category} documents)', fontsize=16)
        plt.xticks(x, batch_sizes)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{category}_combined_time.png'), dpi=300)
        plt.close()
        
        # 2. Combined Throughput Chart
        plt.figure(figsize=(14, 8))
        
        # Get sequential throughput as reference
        if ref_proc in all_results and category in all_results[ref_proc]:
            seq_throughput = [all_results[ref_proc][category]["sequential"][str(b)]["docs_per_second"] for b in batch_sizes]
            plt.bar(x - 1.5*width, seq_throughput, width, label=f'Sequential', color='gray')
        
        # Add MPI throughput for each process count
        for i, n_proc in enumerate(process_counts):
            if n_proc not in all_results or category not in all_results[n_proc]:
                continue
                
            results = all_results[n_proc][category]
            # Adjust position to create grouped bars
            pos = x - 0.5*width + i*width
            throughput = [results["mpi"][str(b)]["docs_per_second"] for b in batch_sizes]
            plt.bar(pos, throughput, width, label=f'{n_proc} processes', color=colors[i])
        
        plt.xlabel('Batch Size', fontsize=14)
        plt.ylabel('Documents per Second', fontsize=14)
        plt.title(f'Throughput Comparison ({category} documents)', fontsize=16)
        plt.xticks(x, batch_sizes)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{category}_combined_throughput.png'), dpi=300)
        plt.close()
        
        # 3. Combined Speedup Chart
        plt.figure(figsize=(14, 8))
        
        # Add reference line for sequential (speedup = 1.0)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Sequential')
        
        # Add MPI speedup for each process count
        for i, n_proc in enumerate(process_counts):
            if n_proc not in all_results or category not in all_results[n_proc]:
                continue
                
            results = all_results[n_proc][category]
            speedups = [results["mpi"][str(b)]["speedup"] for b in batch_sizes]
            plt.plot(batch_sizes, speedups, 'o-', linewidth=2, markersize=8, 
                     label=f'{n_proc} processes', color=colors[i])
        
        plt.xlabel('Batch Size', fontsize=14)
        plt.ylabel('Speedup (vs Sequential)', fontsize=14)
        plt.title(f'Speedup Comparison ({category} documents)', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(batch_sizes)
        plt.savefig(os.path.join(output_dir, f'{category}_combined_speedup.png'), dpi=300)
        plt.close()
        
        # 4. Ideal vs Actual Speedup
        plt.figure(figsize=(14, 8))
        
        # Plot ideal speedup line
        plt.plot(process_counts, process_counts, 'k--', linewidth=2, label='Ideal Linear Speedup')
        
        # Plot actual speedups for each batch size
        markers = ['o', 's', 'd', '^', 'v', '<', '>']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, batch in enumerate(batch_sizes):
            if i >= len(markers) or i >= len(colors):
                continue
                
            actual_speedups = []
            proc_list = []
            
            # Get sequential reference time from first process count results
            for n_proc in process_counts:
                if n_proc not in all_results or category not in all_results[n_proc]:
                    continue
                    
                if str(batch) not in all_results[n_proc][category]["mpi"]:
                    continue
                
                speedup = all_results[n_proc][category]["mpi"][str(batch)]["speedup"]
                actual_speedups.append(speedup)
                proc_list.append(n_proc)
            
            if actual_speedups and proc_list:
                plt.plot(proc_list, actual_speedups, f'{markers[i]}-', 
                         linewidth=2, markersize=8, label=f'Batch size {batch}', color=colors[i])
        
        plt.xlabel('Number of Processes', fontsize=14)
        plt.ylabel('Speedup (vs Sequential)', fontsize=14)
        plt.title(f'Ideal vs. Actual Speedup ({category} documents)', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(process_counts)
        plt.savefig(os.path.join(output_dir, f'{category}_ideal_vs_actual.png'), dpi=300)
        plt.close()
    
    # Create cross-category comparison charts
    print("Generating document size comparison charts...")
    
    # Find common batch size and process count across all categories
    common_batch_size = None
    common_categories = []
    
    # First, determine which categories have data
    for category in categories:
        has_data = True
        for n_proc in process_counts:
            if n_proc not in all_results or category not in all_results[n_proc]:
                has_data = False
                break
        if has_data:
            common_categories.append(category)
    
    if len(common_categories) > 1:  # Only if we have multiple categories with data
        # Find a common batch size
        for category in common_categories:
            category_batch_sizes = set()
            for n_proc in process_counts:
                if category in all_results[n_proc]:
                    batch_sizes = set(all_results[n_proc][category]["mpi"].keys())
                    if not category_batch_sizes:
                        category_batch_sizes = batch_sizes
                    else:
                        category_batch_sizes &= batch_sizes
            
            if common_batch_size is None:
                common_batch_size = category_batch_sizes
            else:
                common_batch_size &= category_batch_sizes
        
        if common_batch_size:
            # Use the middle batch size if available, otherwise the smallest
            common_batch_sizes = sorted([int(b) for b in common_batch_size])
            if len(common_batch_sizes) > 2:
                selected_batch = common_batch_sizes[len(common_batch_sizes) // 2]  # Middle batch size
            else:
                selected_batch = common_batch_sizes[0]  # Smallest batch size
            
            print(f"Using batch size {selected_batch} for cross-category comparison")
            
            # 1. Document Size Impact on Processing Time
            plt.figure(figsize=(14, 8))
            
            # Create a bar grouping for each document size category
            x = np.arange(len(common_categories))
            width = 0.2  # Width of each bar
            
            # Add sequential timing for reference
            ref_proc = process_counts[0]
            seq_times = []
            for category in common_categories:
                if category in all_results[ref_proc] and str(selected_batch) in all_results[ref_proc][category]["sequential"]:
                    seq_times.append(all_results[ref_proc][category]["sequential"][str(selected_batch)]["total_time"])
                else:
                    seq_times.append(0)
            
            # Only plot if we have data
            if any(seq_times):
                plt.bar(x - 1.5*width, seq_times, width, label=f'Sequential', color='gray')
            
            # Add MPI times for each process count
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i, n_proc in enumerate(process_counts):
                times = []
                for category in common_categories:
                    if (category in all_results[n_proc] and 
                        str(selected_batch) in all_results[n_proc][category]["mpi"]):
                        times.append(all_results[n_proc][category]["mpi"][str(selected_batch)]["total_time"])
                    else:
                        times.append(0)
                
                # Only plot if we have data
                if any(times):
                    pos = x - 0.5*width + i*width
                    plt.bar(pos, times, width, label=f'{n_proc} processes', color=colors[i])
            
            plt.xlabel('Document Size Category', fontsize=14)
            plt.ylabel('Processing Time (s)', fontsize=14)
            plt.title(f'Document Size Impact on Processing Time (Batch Size {selected_batch})', fontsize=16)
            plt.xticks(x, [cat.capitalize() for cat in common_categories])
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'size_impact_time.png'), dpi=300)
            plt.close()
            
            # 2. Document Size Impact on Throughput
            plt.figure(figsize=(14, 8))
            
            # Add sequential throughput for reference
            seq_throughput = []
            for category in common_categories:
                if category in all_results[ref_proc] and str(selected_batch) in all_results[ref_proc][category]["sequential"]:
                    seq_throughput.append(all_results[ref_proc][category]["sequential"][str(selected_batch)]["docs_per_second"])
                else:
                    seq_throughput.append(0)
            
            # Only plot if we have data
            if any(seq_throughput):
                plt.bar(x - 1.5*width, seq_throughput, width, label='Sequential', color='gray')
            
            # Add MPI throughput for each process count
            for i, n_proc in enumerate(process_counts):
                throughput = []
                for category in common_categories:
                    if (category in all_results[n_proc] and 
                        str(selected_batch) in all_results[n_proc][category]["mpi"]):
                        throughput.append(all_results[n_proc][category]["mpi"][str(selected_batch)]["docs_per_second"])
                    else:
                        throughput.append(0)
                
                # Only plot if we have data
                if any(throughput):
                    pos = x - 0.5*width + i*width
                    plt.bar(pos, throughput, width, label=f'{n_proc} processes', color=colors[i])
            
            plt.xlabel('Document Size Category', fontsize=14)
            plt.ylabel('Documents per Second', fontsize=14)
            plt.title(f'Document Size Impact on Throughput (Batch Size {selected_batch})', fontsize=16)
            plt.xticks(x, [cat.capitalize() for cat in common_categories])
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'size_impact_throughput.png'), dpi=300)
            plt.close()
            
            # 3. Document Size Impact on Speedup
            plt.figure(figsize=(14, 8))
            
            # Add reference line for sequential (speedup = 1.0)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Sequential')
            
            # Plot speedup for each process count
            for i, n_proc in enumerate(process_counts):
                speedups = []
                for category in common_categories:
                    if (category in all_results[n_proc] and 
                        str(selected_batch) in all_results[n_proc][category]["mpi"]):
                        speedups.append(all_results[n_proc][category]["mpi"][str(selected_batch)]["speedup"])
                    else:
                        speedups.append(0)
                
                # Only plot if we have valid data
                if any(speedups) and all(s > 0 for s in speedups):
                    markers = ['o', 's', 'd']
                    plt.plot(common_categories, speedups, f'{markers[i]}-', linewidth=2, markersize=10,
                            label=f'{n_proc} processes', color=colors[i])
            
            plt.xlabel('Document Size Category', fontsize=14)
            plt.ylabel('Speedup (vs Sequential)', fontsize=14)
            plt.title(f'Document Size Impact on Speedup (Batch Size {selected_batch})', fontsize=16)
            plt.xticks(range(len(common_categories)), [cat.capitalize() for cat in common_categories])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(output_dir, 'size_impact_speedup.png'), dpi=300)
            plt.close()
            
            # 4. Document Size Impact on Efficiency
            plt.figure(figsize=(14, 8))
            
            # Plot efficiency for each process count
            for i, n_proc in enumerate(process_counts):
                efficiencies = []
                for category in common_categories:
                    if (category in all_results[n_proc] and 
                        str(selected_batch) in all_results[n_proc][category]["mpi"]):
                        efficiencies.append(all_results[n_proc][category]["mpi"][str(selected_batch)]["efficiency"])
                    else:
                        efficiencies.append(0)
                
                # Only plot if we have valid data
                if any(efficiencies) and all(e > 0 for e in efficiencies):
                    markers = ['o', 's', 'd']
                    plt.plot(common_categories, efficiencies, f'{markers[i]}-', linewidth=2, markersize=10,
                            label=f'{n_proc} processes', color=colors[i])
            
            plt.xlabel('Document Size Category', fontsize=14)
            plt.ylabel('Parallel Efficiency', fontsize=14)
            plt.title(f'Document Size Impact on Efficiency (Batch Size {selected_batch})', fontsize=16)
            plt.xticks(range(len(common_categories)), [cat.capitalize() for cat in common_categories])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(output_dir, 'size_impact_efficiency.png'), dpi=300)
            plt.close()
    
    print(f"All combined comparison charts saved to {output_dir} directory")


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running with {size} MPI processes")
    
    # Only rank 0 parses arguments
    if rank == 0:
        parser = argparse.ArgumentParser(description='MPI Benchmark for Parallel Document Embedding')
        parser.add_argument('--docs_per_category', type=int, default=128, 
                           help='Number of documents per size category (small, medium, large)')
        parser.add_argument('--min_length', type=int, default=20, help='Minimum document length (words)')
        parser.add_argument('--max_length', type=int, default=1000, help='Maximum document length (words)')
        parser.add_argument('--batch_sizes', type=str, default='16,64,256', 
                           help='Comma-separated batch sizes')
        parser.add_argument('--output_dir', type=str, default='mpi_benchmark_results', 
                           help='Output directory for results')
        
        args = parser.parse_args()
        
        # Parse batch sizes
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
        
        print(f"Generating balanced document datasets...")
        gen_start = time.time()
        data_categories = generate_balanced_test_data(
            docs_per_category=args.docs_per_category,
            min_length=args.min_length,
            max_length=args.max_length
        )
        gen_end = time.time()
        print(f"Document generation took {gen_end - gen_start:.2f} seconds")
    else:
        data_categories = None
        batch_sizes = None
        args = None
    
    # Broadcast data to all processes
    bcast_start = time.time()
    data_categories = comm.bcast(data_categories, root=0)
    batch_sizes = comm.bcast(batch_sizes, root=0)
    bcast_end = time.time()
    
    if rank == 0:
        print(f"Broadcasting data to all processes took {bcast_end - bcast_start:.2f} seconds")
    
    if rank == 0:
        print("Running MPI benchmarks...")
    
    # Run benchmark
    bench_start = time.time()
    results = run_mpi_benchmark(data_categories, batch_sizes)
    bench_end = time.time()
    
    if rank == 0:
        print(f"Benchmark execution took {bench_end - bench_start:.2f} seconds")
    
    # Only rank 0 saves results and plots
    if rank == 0:
        # Save results to JSON
        os.makedirs(args.output_dir, exist_ok=True)
        
        save_start = time.time()
        with open(os.path.join(args.output_dir, 'mpi_benchmark_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        save_end = time.time()
        print(f"Saving results took {save_end - save_start:.2f} seconds")
        
        print("Plotting results...")
        plot_start = time.time()
        plot_mpi_results(results, args.output_dir)
        plot_end = time.time()
        print(f"Plotting results took {plot_end - plot_start:.2f} seconds")
        
        print("Done!")


if __name__ == "__main__":
    overall_start = time.time()
    main()
    overall_end = time.time()
    
    # Only rank 0 prints the final timing
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print(f"Total execution time: {overall_end - overall_start:.2f} seconds")