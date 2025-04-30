import matplotlib.pyplot as plt
import json
import numpy as np
import os
import re

def load_multiple_data(json_paths):
    """Load benchmark data from multiple JSON files"""
    combined_results = {}
    
    for json_path in json_paths:
        # Extract the number of processes from the path using regex
        match = re.search(r'n(\d+)', json_path)
        if match:
            num_processes = int(match.group(1))
        else:
            # Default fallback if pattern not found
            num_processes = 2  # Assuming default is 2 processes
            print(f"Warning: Could not extract process count from {json_path}, assuming {num_processes}")
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Store in combined_results with num_processes as the key
        combined_results[num_processes] = data
    
    return combined_results

def generate_advanced_charts(combined_results, output_dir="final_charts"):
    """Generate advanced charts comparing document sizes, batch sizes, and processes"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all available data
    processes = sorted(combined_results.keys())
    doc_categories = list(combined_results[processes[0]].keys())  # Assuming consistent across files
    
    # Get all batch sizes (assuming they're the same across all process configurations)
    reference_data = combined_results[processes[0]]["small"]["sequential"]
    batch_sizes = sorted([int(b) for b in reference_data.keys()])
    
    print(f"Generating charts for processes: {processes}")
    print(f"Document categories: {doc_categories}")
    print(f"Batch sizes: {batch_sizes}")
    
    # Set up color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(processes)))
    
    # 1. Processing Time by Number of Processes (vertical subplots)
    # Select representative batch sizes
    common_batch = "64"
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    # Track max_time for consistent y-axis limits
    max_time = 0
    
    for doc_idx, category in enumerate(doc_categories):
        ax = axes[doc_idx]
        
        # Calculate sequential reference time
        seq_times = {}
        mpi_times = {}
        
        for proc in processes:
            data = combined_results[proc][category]
            seq_times[proc] = data["sequential"][common_batch]["total_time"]
            mpi_times[proc] = data["mpi"][common_batch]["total_time"]
            max_time = max(max_time, seq_times[proc], mpi_times[proc])
        
        # Set up x positions
        x = np.arange(len(processes))
        width = 0.35
        
        # Plot
        ax.bar(x - width/2, [seq_times[p] for p in processes], width, 
              label='Sequential', color='#1f77b4', alpha=0.7)
        ax.bar(x + width/2, [mpi_times[p] for p in processes], width, 
              label='MPI', color='#ff7f0e')
        
        # Add text labels
        for i, proc in enumerate(processes):
            ax.text(i - width/2, seq_times[proc] + 0.5, f"{seq_times[proc]:.1f}s", 
                  ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, mpi_times[proc] + 0.5, f"{mpi_times[proc]:.1f}s", 
                  ha='center', va='bottom', fontsize=9)
        
        # Set labels
        ax.set_ylabel('Processing Time (s)', fontsize=14)
        ax.set_title(f'Processing Time: {category.capitalize()} Documents, Batch Size {common_batch}', fontsize=16)
        
        # Add legend only to the first subplot
        if doc_idx == 0:
            ax.legend(fontsize=12, loc='upper right')
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set common x-axis label
    axes[-1].set_xlabel('Number of Processes', fontsize=14)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(processes)
    
    # Set common y-axis limit with padding
    for ax in axes:
        ax.set_ylim(0, max_time * 1.2)
    
    # Add overall title
    fig.suptitle('Processing Time Analysis by Document Category', fontsize=20, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Make room for the suptitle
    plt.savefig(os.path.join(output_dir, f'proc_time_all_batch{common_batch}.png'), dpi=300)
    plt.close()
    
    # 2. Speedup Comparison Across All Parameters - Vertical Subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    
    # We'll create a grouped bar chart for each document size, arranged vertically
    x_base = np.arange(len(batch_sizes))
    bar_width = 0.8 / len(processes)  # Width of bars
    
    # Set up y-limit tracking
    max_speedup = 0
    
    # Create subplots for each document size
    for doc_idx, category in enumerate(doc_categories):
        ax = axes[doc_idx]
        
        # Create bars for each process count
        for proc_idx, proc in enumerate(processes):
            speedups = []
            for batch in batch_sizes:
                data = combined_results[proc][category]
                seq_time = data["sequential"][str(batch)]["total_time"]
                mpi_time = data["mpi"][str(batch)]["total_time"]
                speedup = seq_time / mpi_time
                speedups.append(speedup)
                max_speedup = max(max_speedup, speedup)
            
            # Calculate position for this group of bars
            positions = x_base + (proc_idx - len(processes)/2 + 0.5) * bar_width
            
            # Plot bars
            bars = ax.bar(positions, speedups, bar_width * 0.9, 
                    label=f'{proc} processes',
                    color=colors[proc_idx])
            
            # Add text for the maximum speedup
            max_proc_speedup = max(speedups)
            max_idx = speedups.index(max_proc_speedup)
            ax.text(positions[max_idx], max_proc_speedup + 0.1, 
                  f"{max_proc_speedup:.2f}x", ha='center', va='bottom', fontsize=10)
        
        # Add a horizontal line at y=1
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        
        # Set labels and title for this subplot
        ax.set_ylabel('Speedup (vs Sequential)', fontsize=14)
        ax.set_title(f'Speedup Analysis: {category.capitalize()} Documents', fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend to first subplot only
        if doc_idx == 0:
            ax.legend(fontsize=12, loc='upper right')
    
    # Add common x-axis label to the bottom subplot
    axes[-1].set_xlabel('Batch Size', fontsize=14)
    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(batch_sizes)
    
    # Add a main title for the entire figure
    fig.suptitle('Comprehensive Speedup Analysis by Document Size', fontsize=20, y=0.98)
    
    # Set common y-limits with some padding
    for ax in axes:
        ax.set_ylim(0, max_speedup * 1.2)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Make room for the suptitle
    plt.savefig(os.path.join(output_dir, 'comprehensive_speedup.png'), dpi=300)
    plt.close()
    
    # 3. Efficiency Heatmaps for document sizes in vertical layout
    fig, axes = plt.subplots(3, 1, figsize=(14, 20))
    
    for doc_idx, category in enumerate(doc_categories):
        ax = axes[doc_idx]
        
        # Create a matrix of efficiencies: rows=processes, cols=batch_sizes
        efficiency_matrix = np.zeros((len(processes), len(batch_sizes)))
        
        for i, proc in enumerate(processes):
            for j, batch in enumerate(batch_sizes):
                data = combined_results[proc][category]
                # Calculate efficiency directly
                seq_time = data["sequential"][str(batch)]["total_time"]
                mpi_time = data["mpi"][str(batch)]["total_time"]
                speedup = seq_time / mpi_time
                efficiency = speedup / proc
                efficiency_matrix[i, j] = efficiency
        
        # Create heatmap
        im = ax.imshow(efficiency_matrix, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax, label='Parallel Efficiency')
        
        # Set labels
        if doc_idx == len(doc_categories) - 1:  # Only bottom plot gets x-axis label
            ax.set_xlabel('Batch Size', fontsize=14)
        ax.set_ylabel('Number of Processes', fontsize=14)
        ax.set_title(f'Parallel Efficiency: {category.capitalize()} Documents', fontsize=16)
        
        # Set ticks
        ax.set_xticks(np.arange(len(batch_sizes)))
        ax.set_xticklabels(batch_sizes)
        ax.set_yticks(np.arange(len(processes)))
        ax.set_yticklabels(processes)
        
        # Add text annotations
        for i in range(len(processes)):
            for j in range(len(batch_sizes)):
                ax.text(j, i, f'{efficiency_matrix[i, j]:.2f}', 
                       ha='center', va='center', 
                       color='white' if efficiency_matrix[i, j] < 0.5 else 'black')
    
    # Add overall title
    fig.suptitle('Parallel Efficiency Analysis by Document Category', fontsize=20, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Make room for the suptitle
    plt.savefig(os.path.join(output_dir, 'efficiency_heatmaps_all.png'), dpi=300)
    plt.close()
    
    # 4. Throughput by Batch Size for each document category (vertical subplots)
    fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    # Track max throughput for consistent y-axis scaling
    max_throughput = 0
    
    for doc_idx, category in enumerate(doc_categories):
        ax = axes[doc_idx]
        
        # Set up x-axis for batch sizes
        x = np.arange(len(batch_sizes))
        
        # Plot sequential as reference
        seq_throughput = []
        for batch in batch_sizes:
            seq_throughput.append(combined_results[processes[0]][category]["sequential"][str(batch)]["docs_per_second"])
            max_throughput = max(max_throughput, combined_results[processes[0]][category]["sequential"][str(batch)]["docs_per_second"])
        
        ax.plot(x, seq_throughput, 'k-', marker='o', linewidth=2, markersize=8, label='Sequential')
        
        # Plot MPI results for each process count
        for i, proc in enumerate(processes):
            mpi_throughput = []
            for batch in batch_sizes:
                throughput = combined_results[proc][category]["mpi"][str(batch)]["docs_per_second"]
                mpi_throughput.append(throughput)
                max_throughput = max(max_throughput, throughput)
            
            ax.plot(x, mpi_throughput, color=colors[i % len(colors)], marker='s', linewidth=2, 
                   markersize=8, label=f'MPI ({proc} processes)')
        
        # Set labels and title
        ax.set_ylabel('Documents per Second', fontsize=14)
        ax.set_title(f'Throughput by Batch Size: {category.capitalize()} Documents', fontsize=16)
        
        # Add legend only to the first subplot
        if doc_idx == 0:
            ax.legend(fontsize=12, loc='upper right')
            
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set common x-axis label
    axes[-1].set_xlabel('Batch Size', fontsize=14)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(batch_sizes)
    
    # Set common y-axis limits
    for ax in axes:
        ax.set_ylim(0, max_throughput * 1.2)
    
    # Add overall title
    fig.suptitle('Throughput Analysis by Document Category', fontsize=20, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Make room for the suptitle
    plt.savefig(os.path.join(output_dir, 'throughput_by_batch_size_all.png'), dpi=300)
    plt.close()
    
    # 5. Speedup vs. Batch Size Line Chart (vertical subplots)
    fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    max_speedup = 0
    
    for doc_idx, category in enumerate(doc_categories):
        ax = axes[doc_idx]
        
        for i, proc in enumerate(processes):
            speedups = []
            for batch in batch_sizes:
                data = combined_results[proc][category]
                seq_time = data["sequential"][str(batch)]["total_time"]
                mpi_time = data["mpi"][str(batch)]["total_time"]
                speedup = seq_time / mpi_time
                speedups.append(speedup)
                max_speedup = max(max_speedup, speedup)
            
            ax.plot(batch_sizes, speedups, marker='o', linewidth=2, markersize=8, 
                   label=f'{proc} processes', color=colors[i])
        
        # Add horizontal line at y=1
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='No speedup' if doc_idx == 0 else "")
        
        ax.set_ylabel('Speedup (vs Sequential)', fontsize=14)
        ax.set_title(f'Speedup vs. Batch Size: {category.capitalize()} Documents', fontsize=16)
        
        # Only add legend to the first subplot
        if doc_idx == 0:
            ax.legend(fontsize=12, loc='upper right')
            
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set common x-axis label for the bottom subplot
    axes[-1].set_xlabel('Batch Size', fontsize=14)
    
    # Set common y-limits
    for ax in axes:
        ax.set_ylim(0, max_speedup * 1.2)
    
    # Add overall title
    fig.suptitle('Speedup vs. Batch Size Analysis by Document Category', fontsize=20, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Make room for the suptitle
    plt.savefig(os.path.join(output_dir, 'speedup_vs_batchsize_all.png'), dpi=300)
    plt.close()
    
    # 6. Ideal vs. Actual Speedup
    plt.figure(figsize=(14, 8))
    
    # Choose a specific batch size and document category for clarity
    batch = "32"
    category = "medium"
    
    actual_speedups = []
    for proc in processes:
        data = combined_results[proc][category]
        seq_time = data["sequential"][batch]["total_time"]
        mpi_time = data["mpi"][batch]["total_time"]
        speedup = seq_time / mpi_time
        actual_speedups.append(speedup)
    
    # Ideal speedup is linear with number of processes
    ideal_speedups = processes.copy()  # Assuming processes is already a list
    
    plt.plot(processes, ideal_speedups, 'r--', linewidth=2, label='Ideal Speedup')
    plt.plot(processes, actual_speedups, 'bo-', linewidth=2, markersize=8, label='Actual Speedup')
    
    # Add text labels for actual speedups
    for i, proc in enumerate(processes):
        plt.text(proc, actual_speedups[i] + 0.2, f"{actual_speedups[i]:.2f}x", 
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Number of Processes', fontsize=14)
    plt.ylabel('Speedup (vs Sequential)', fontsize=14)
    plt.title(f'Ideal vs Actual Speedup ({category.capitalize()} Documents, Batch Size {batch})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ideal_vs_actual_speedup_{category}_batch{batch}.png'), dpi=300)
    plt.close()
    
    print(f"Generated advanced charts saved to {output_dir}")


if __name__ == "__main__":
    # Specify the paths to JSON files
    json_files = [
        "C:/Users/tmehd/Desktop/NC-State/CSC548/RAG-Streaming/mpi_results/n2/mpi_benchmark_results.json",
        "C:/Users/tmehd/Desktop/NC-State/CSC548/RAG-Streaming/mpi_results/n4/mpi_benchmark_results.json",
        "C:/Users/tmehd/Desktop/NC-State/CSC548/RAG-Streaming/mpi_results/n8/mpi_benchmark_results.json"
    ]
    
    # Load the data from multiple files
    combined_results = load_multiple_data(json_files)
    
    # Generate advanced charts
    generate_advanced_charts(combined_results, "final_charts")