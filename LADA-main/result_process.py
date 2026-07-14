import os
import re
import yaml
import numpy as np
import argparse


def read_yaml_config(yaml_path):
    """Read YAML configuration file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['dataset_sequence']


def extract_task_accuracies(log_file_path, num_tasks):
    """Extract Task accuracies from log file"""
    if not os.path.exists(log_file_path):
        return None
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Match Task accuracy pattern
    pattern = r'\* Task (\d+) Accuracy: ([\d.]+)%'
    matches = re.findall(pattern, content)
    
    if not matches:
        return None
    
    # Get the last occurrence of each Task's result
    task_accuracies = {}
    lines = content.split('\n')
    
    for line in reversed(lines):
        match = re.search(pattern, line)
        if match:
            task_id = int(match.group(1))
            accuracy = float(match.group(2))
            if task_id not in task_accuracies:
                task_accuracies[task_id] = accuracy
    
    if len(task_accuracies) == num_tasks:
        result = []
        for i in range(num_tasks):
            result.append(task_accuracies.get(i, 0.0))
        return result
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Quick read and display log results")
    parser.add_argument("--data", "-d", type=str, 
                        help="Path to TAIL yaml config file")
    parser.add_argument("--output_dir", type=str, 
                        help="Directory containing log files")
    
    args = parser.parse_args()
    
    # Configuration
    yaml_path = os.path.join("./configs/data", args.data + ".yaml")
    log_dir = os.path.join("./output", args.output_dir)
    
    # Read dataset sequence
    try:
        dataset_sequence = read_yaml_config(yaml_path)
    except Exception as e:
        print(f"Error reading YAML file {yaml_path}: {e}")
        return None
    
    # Store results
    results = []
    dataset_names = []
    missing_files = []
    num_tasks = len(dataset_sequence)
    
    for dataset in dataset_sequence:
        log_file = os.path.join(log_dir, f"log_{dataset}.txt")
        accuracies = extract_task_accuracies(log_file, num_tasks)
        
        if accuracies is not None:
            results.append(accuracies)
            dataset_names.append(dataset)
        else:
            missing_files.append(dataset)
    
    if results:
        # Convert to numpy array
        result_matrix = np.array(results)
        
        # Create result file path
        result_file_path = os.path.join(log_dir, "result.txt")
        
        # Define output function to print to console and write to file
        def output_line(line, file_handle=None):
            print(line)
            if file_handle:
                file_handle.write(line + "\n")
        
        # Open file for writing
        with open(result_file_path, 'w') as f:
            output_line("=" * 75, f)
            line = f"{'Dataset':<15}"
            for dataset in dataset_sequence:
                dataset_short = dataset[:3] + "."
                line += f"{dataset_short:<6}"
            output_line(line, f)
            
            output_line("-" * 75, f)
            
            # Print data
            for i, dataset in enumerate(dataset_names):
                line = f"{dataset:<15}"
                for j in range(len(dataset_sequence)):
                    if j < len(result_matrix[i]):
                        line += f"{result_matrix[i][j]:<6.1f}"
                    else:
                        line += f"{'N/A':<6}"
                output_line(line, f)
            
            # Output missing dataset information
            for file in missing_files:
                output_line(f"{file:<15}FILE NOT FOUND OR NO DATA !!!", f)
        
            # Calculate transfer, average, last metrics (only when it's a square matrix)
            if result_matrix.shape[0] == result_matrix.shape[1] and result_matrix.shape[0] == len(dataset_sequence):
                output_line("\n" + "=" * 75, f)
                output_line("EVALUATION METRICS", f)
                output_line("=" * 75, f)
                
                # Calculate metrics
                transfer_metrics = []
                average_metrics = []
                last_metrics = []
                
                for k in range(len(dataset_sequence)):
                    # Transfer: first dataset has no transfer, calculate from second dataset
                    if k == 0:
                        transfer_metrics.append("N/A")
                    else:
                        # Calculate mean of the first k rows in this column
                        transfer_sum = sum(result_matrix[j][k] for j in range(k))
                        transfer_avg = transfer_sum / k
                        transfer_metrics.append(f"{transfer_avg:.1f}")
                    
                    # Average: mean of each column
                    column_avg = sum(result_matrix[j][k] for j in range(len(dataset_names))) / len(dataset_names)
                    average_metrics.append(f"{column_avg:.1f}")
                    
                    # Last: value from the last row
                    last_metrics.append(f"{result_matrix[-1][k]:.1f}")
                
                # Output metrics
                line = f"{'Transfer':<15}"
                for metric in transfer_metrics:
                    line += f"{metric:<6}"
                output_line(line, f)
                
                line = f"{'Average':<15}"
                for metric in average_metrics:
                    line += f"{metric:<6}"
                output_line(line, f)
                
                line = f"{'Last':<15}"
                for metric in last_metrics:
                    line += f"{metric:<6}"
                output_line(line, f)
                
                # Calculate mean of each metric
                output_line("=" * 75, f)
                
                # Transfer mean: exclude N/A, only calculate valid values
                transfer_values = [float(m) for m in transfer_metrics if m != "N/A"]
                transfer_mean = sum(transfer_values) / len(transfer_values) if transfer_values else 0
                
                # Average mean: all values
                average_values = [float(m) for m in average_metrics]
                average_mean = sum(average_values) / len(average_values)
                
                # Last mean: all values
                last_values = [float(m) for m in last_metrics]
                last_mean = sum(last_values) / len(last_values)
                
                output_line(f"Transfer Mean: {transfer_mean:.1f}", f)
                output_line(f"Average Mean:  {average_mean:.1f}", f)
                output_line(f"Last Mean:     {last_mean:.1f}", f)
            
            output_line("=" * 75, f)
        return result_matrix, dataset_names
    else:
        print("No valid results found!")
        if missing_files:
            print(f"Missing files or no data: {', '.join(missing_files)}")
        return None
    

if __name__ == "__main__":
    main()
