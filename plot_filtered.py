import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import sys
import matplotlib.axes

# Default values from the problem description
K_DEFAULT = 300
Q_DEFAULT = 10000

def load_indices(file_path: Path, q: int, k: int) -> np.ndarray | None:
    """Loads k*q uint32_t indices from a binary file and reshapes to (q, k)."""
    try:
        data = np.fromfile(file_path, dtype=np.uint32)
        if data.size != q * k:
            print(f"Warning: Expected {q * k} elements in {file_path}, found {data.size}. Skipping this file.", file=sys.stderr)
            return None
        return data.reshape((q, k))
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}. Skipping.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading indices from {file_path}: {e}", file=sys.stderr)
        return None

def load_stats_file(filepath: Path, q_val: int) -> dict | None:
    """Loads custom statistics from a stats_filtered_out_* file."""
    try:
        with open(filepath, 'rb') as f:
            magic_number_arr = np.fromfile(f, dtype=np.uint32, count=1)
            if not magic_number_arr.size: # Empty file or could not read
                # print(f"  Warning: Stats file {filepath.name} is empty or unreadable at start.", file=sys.stderr)
                return {} # Treat as no stats found
            # magic_number = magic_number_arr[0]
            # print(f"  Stats file {filepath.name}: Magic number = {magic_number}") # Optional

            data_offset_arr = np.fromfile(f, dtype=np.uint32, count=1)
            if not data_offset_arr.size: return {}
            data_offset = data_offset_arr[0]

            num_fields_arr = np.fromfile(f, dtype=np.uint32, count=1)
            if not num_fields_arr.size: return {}
            num_fields = int(num_fields_arr[0])

            if num_fields == 0:
                # print(f"  Stats file {filepath.name} reports 0 custom fields. No stats data loaded.", file=sys.stderr)
                return {}

            field_sizes = np.fromfile(f, dtype=np.uint32, count=num_fields)
            if len(field_sizes) != num_fields:
                # print(f"  Error: Truncated field_sizes in {filepath.name}", file=sys.stderr)
                return None # Indicates a parsing error
            if not np.all(field_sizes == 4):
                print(f"  Warning: Stats file {filepath.name} has field_sizes not all uint32 (4 bytes). Actual: {field_sizes}. Assuming uint32 data.", file=sys.stderr)

            current_pos = f.tell()
            desc_block_size = data_offset - current_pos
            if desc_block_size < 0:
                print(f"  Error: data_offset ({data_offset}) is before end of header ({current_pos}) in {filepath.name}", file=sys.stderr)
                return None

            desc_bytes = f.read(desc_block_size)
            try:
                # Decode, strip NUL bytes from the end, then split by newline, then strip individual names.
                field_descriptions_str = desc_bytes.decode('utf-8', errors='replace').rstrip('\x00')
                field_names = [name.strip() for name in field_descriptions_str.split('\n') if name.strip()]
            except Exception as e:
                print(f"  Error decoding/parsing field descriptions in {filepath.name}: {e}", file=sys.stderr)
                return None


            if len(field_names) != num_fields:
                print(f"  Critical: Mismatch between num_fields in header ({num_fields}) and parsed names ({len(field_names)}) for {filepath.name}. Names: {field_names}", file=sys.stderr)
                return None

            f.seek(data_offset)

            expected_data_elements = q_val * num_fields
            if q_val == 0 and num_fields > 0:
                 # print(f"  Warning: Q_val is 0 for {filepath.name}, but num_fields is {num_fields}. No data to average.", file=sys.stderr)
                 return {name: float('nan') for name in field_names} # Or 0.0
            elif expected_data_elements == 0 :
                 return {}


            stats_data_flat = np.fromfile(f, dtype=np.uint32, count=expected_data_elements)
            if stats_data_flat.size != expected_data_elements:
                print(f"  Error: Expected {expected_data_elements} data elements in {filepath.name}, found {stats_data_flat.size}", file=sys.stderr)
                return None

            if q_val == 0 : # Should have been caught by expected_data_elements == 0 if num_fields > 0
                return {name: float('nan') for name in field_names}

            stats_data_reshaped = stats_data_flat.reshape((q_val, num_fields))
            average_stats = np.mean(stats_data_reshaped, axis=0)

            stats_dict = {field_names[i]: average_stats[i] for i in range(num_fields)}
            return stats_dict

    except FileNotFoundError:
        # This is common if not all runs produce stats files; treat as non-fatal.
        # print(f"  Note: Stats file not found {filepath}. No custom stats for this point.", file=sys.stderr)
        return {} # Return empty dict, indicating stats were looked for but not found.
    except Exception as e:
        print(f"  Error loading stats from {filepath.name}: {e}", file=sys.stderr)
        return None # Indicates a more significant error during parsing.

def compute_recall_1_at_K(naive_indices: np.ndarray, filtered_indices: np.ndarray, q_val: int, k_val: int) -> float:
    """
    Computes 1@K recall: E[X^1 intersect Y / 1].
    X_i^1 is {naive[i][0]}, Y_i is filtered_indices[i].
    """
    if naive_indices.shape[0] != q_val or filtered_indices.shape[0] != q_val:
        print("Warning: Mismatch in query count for recall calculation.", file=sys.stderr)
        return 0.0
    if naive_indices.shape[1] < 1 or filtered_indices.shape[1] < k_val : # filtered_indices should have at least k_val, naive at least 1
         print(f"Warning: Not enough results for recall_1_at_K. Naive shape {naive_indices.shape}, Filtered shape {filtered_indices.shape}, K={k_val}", file=sys.stderr)
         return 0.0


    successful_queries = 0
    for i in range(q_val):
        ground_truth_top_1 = naive_indices[i, 0]
        # Using a set for faster 'in' check
        filtered_set_for_query = set(filtered_indices[i, :])
        if ground_truth_top_1 in filtered_set_for_query:
            successful_queries += 1

    return successful_queries / q_val if q_val > 0 else 0.0

def compute_recall_K_at_K(naive_indices: np.ndarray, filtered_indices: np.ndarray, q_val: int, k_val: int) -> float:
    """
    Computes K@K recall: E[X intersect Y / K].
    X_i is naive_indices[i], Y_i is filtered_indices[i].
    """
    if naive_indices.shape[0] != q_val or filtered_indices.shape[0] != q_val:
        print("Warning: Mismatch in query count for recall calculation.", file=sys.stderr)
        return 0.0
    if naive_indices.shape[1] < k_val or filtered_indices.shape[1] < k_val:
         print(f"Warning: Not enough results for recall_K_at_K. Naive shape {naive_indices.shape}, Filtered shape {filtered_indices.shape}, K={k_val}", file=sys.stderr)
         return 0.0

    total_normalized_intersection_size = 0
    for i in range(q_val):
        # Consider all K ground truth elements for this query
        ground_truth_set = set(naive_indices[i, :k_val]) # Ensure we take K elements from ground truth
        filtered_set = set(filtered_indices[i, :k_val])   # And K elements from filtered results

        intersection_size = len(ground_truth_set.intersection(filtered_set))
        total_normalized_intersection_size += intersection_size / k_val if k_val > 0 else 0.0

    return total_normalized_intersection_size / q_val if q_val > 0 else 0.0

def process_naive(naive_filepath: Path) -> dict:
    """
    Parses the stdout for naive (prefilter).
    """
    # Regex patterns
    latency_regex = re.compile(r"avg filtered query latency: (\d+\.?\d*) ms")
    # Example: filtered results saved to filtered_out_hnsw_M16_efC128_0.125
    # Group 1: index_name (e.g., hnsw_M16_efC128)
    # Group 2: prob_str (e.g., 0.125)
    results_regex = re.compile(r"filtered results saved to filtered_naive_(\d+\.\d+)")

    collected_data = defaultdict(list)
    current_latency = None

    try:
        with open(naive_filepath, 'r') as f:
            for line in f:
                latency_match = latency_regex.search(line)
                if latency_match:
                    current_latency = float(latency_match.group(1))
                    continue

                results_match = results_regex.search(line)
                if results_match:
                    prob_str = results_match.group(1)

                    if current_latency is None:
                        print(f"Warning: Found results line for naive_{prob_str} but no preceding latency. Skipping.", file=sys.stderr)
                        continue

                    recall_1_k = 1.0
                    recall_k_k = 1.0

                    data_point = {
                        'prob': float(prob_str),
                        'latency': current_latency,
                        'recall_1_k': recall_1_k,
                        'recall_k_k': recall_k_k,
                    }
                    collected_data['naive'].append(data_point)

                    current_latency = None # Reset latency, expect a new one for the next entry

    except FileNotFoundError:
        print(f"Error: stdout log file not found at {naive_filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing {naive_filepath}: {e}", file=sys.stderr)
        sys.exit(1)

    return collected_data


def process_data(stdout_filepath: Path, data_dir: Path, q_val: int, k_val: int) -> dict:
    """
    Parses the stdout log, loads data, computes metrics, and returns structured data for plotting.
    """
    # Regex patterns
    latency_regex = re.compile(r"avg filtered query latency: (\d+\.?\d*) ms")
    # Example: filtered results saved to filtered_out_hnsw_M16_efC128_0.125
    # Group 1: index_name (e.g., hnsw_M16_efC128)
    # Group 2: prob_str (e.g., 0.125)
    results_regex = re.compile(r"filtered results saved to filtered_out_(.+)_(\d+\.\d+)")

    collected_data = defaultdict(list)
    current_latency = None

    try:
        with open(stdout_filepath, 'r') as f:
            for line in f:
                latency_match = latency_regex.search(line)
                if latency_match:
                    current_latency = float(latency_match.group(1))
                    continue

                results_match = results_regex.search(line)
                if results_match:
                    index_name = results_match.group(1)
                    prob_str = results_match.group(2)

                    if current_latency is None:
                        print(f"Warning: Found results line for {index_name}_{prob_str} but no preceding latency. Skipping.", file=sys.stderr)
                        continue

                    naive_filename = f"filtered_naive_{prob_str}"
                    filtered_filename = f"filtered_out_{index_name}_{prob_str}"
                    stats_filename = f"stats_filtered_out_{index_name}_{prob_str}"

                    naive_filepath = data_dir / naive_filename
                    filtered_filepath = data_dir / filtered_filename
                    stats_filepath = data_dir / stats_filename

                    print(f"Processing: Index={index_name}, Prob={prob_str}, Latency={current_latency}ms")
                    print(f"  Naive file: {naive_filepath}")
                    print(f"  Filtered file: {filtered_filepath}")
                    print(f"  Stats file: {stats_filename}")

                    naive_indices = load_indices(naive_filepath, q_val, k_val)
                    if naive_indices is None:
                        print(f"Could not load naive indices for {prob_str}. Skipping data point.", file=sys.stderr)
                        current_latency = None # Reset for next valid pair
                        continue

                    filtered_indices = load_indices(filtered_filepath, q_val, k_val)
                    if filtered_indices is None:
                        print(f"Could not load filtered indices for {index_name}_{prob_str}. Skipping data point.", file=sys.stderr)
                        current_latency = None # Reset for next valid pair
                        continue

                    recall_1_k = compute_recall_1_at_K(naive_indices, filtered_indices, q_val, k_val)
                    recall_k_k = compute_recall_K_at_K(naive_indices, filtered_indices, q_val, k_val)

                    avg_stats = load_stats_file(stats_filepath, q_val)
                    # avg_stats will be {} if file not found/empty, or None on error.
                    # If None, we might choose to skip the point or proceed without stats.
                    # Current load_stats_file returns {} for not-found/empty, None for parse errors.
                    if avg_stats is None: # Serious error parsing stats file
                        print(f"Warning: Failed to parse stats file {stats_filepath.name}. Proceeding without these stats for {index_name}_{prob_str}.", file=sys.stderr)
                        avg_stats = {} # Treat as no stats available

                    data_point = {
                        'prob': float(prob_str),
                        'latency': current_latency,
                        'recall_1_k': recall_1_k,
                        'recall_k_k': recall_k_k,
                        'k_val_at_time_of_calc': k_val # Store K used for this calculation
                    }
                    data_point.update(avg_stats)
                    collected_data[index_name].append(data_point)

                    current_latency = None # Reset latency, expect a new one for the next entry

    except FileNotFoundError:
        print(f"Error: stdout log file not found at {stdout_filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing {stdout_filepath}: {e}", file=sys.stderr)
        sys.exit(1)

    return collected_data

# --- Accessor functions for plotting data ---
AXIS_META = {
    'prob': {
        'label': 'Selectivity',
    },
    'latency': {
        'label': 'Latency (ms)',
    },
    'recall_1_k': {
        'label': 'Recall-1@<K>',
    },
    'recall_k_k': {
        'label': 'Recall-<K>@<K>',
    },
}

def generate_plot(plot_data_map: dict, x_axis_key: str, y_axis_keys: list[str], y_label: str | None, k_val_for_label: int, save: bool):
    """Generates and saves/shows the plot."""
    if not plot_data_map:
        print("No data to plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    if len(y_axis_keys) == 2:
        ax2: matplotlib.axes.Axes | None = ax.twinx() # type: ignore
    else:
        ax2 = None

    linestyles = ['-', '--', ':', '-.']

    x_meta = AXIS_META.get(x_axis_key) or {}

    y_metas = [AXIS_META.get(key, {'label': key}) for key in y_axis_keys]
    if y_label == None:
        y_label = ' / '.join(m['label'].replace('<K>', str(k_val_for_label)) for m in y_metas)

    axis_keys = [x_axis_key] + y_axis_keys

    for index_name, data_points_list in plot_data_map.items():
        if not data_points_list:
            continue

        # Sort points by the x-axis value for a clean line plot
        # It's possible k_val changed if process_data was called multiple times with different k_vals
        # but for a single run, data_point['k_val_at_time_of_calc'] will be args.k_val
        # We use k_val_for_label from args for consistent plot labeling.

        # Filter out points that might not have the required keys (e.g., if metas fail)
        valid_points = []
        for dp in data_points_list:
            if not all(key in dp for key in axis_keys):
                continue
            valid_points.append(dp)

        if not valid_points:
            continue

        valid_points.sort(key=lambda p: p['prob'])

        for i, y_axis_key in enumerate(y_axis_keys):
            x_coords = [p[x_axis_key] for p in valid_points]
            y_coords = [p[y_axis_key] for p in valid_points]

            plot_ax = ax
            if i == 1 and ax2:
                plot_ax = ax2

            plot_ax.plot(x_coords, y_coords, marker='o', linestyle=linestyles[i % len(linestyles)], label=index_name)
            for p in valid_points:
                prob = p['prob']
                if x_axis_key != 'prob':
                    plot_ax.annotate(str(prob), (p[x_axis_key], p[y_axis_key]))

    # --- Plot labels and title ---
    x_label = x_meta.get('label', x_axis_key).replace('<K>', str(k_val_for_label))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(f"{y_label} vs. {x_label}")

    if ax2:
        ax.set_ylabel(y_metas[0]['label'].replace('<K>', str(k_val_for_label)))
        ax2.set_ylabel(y_metas[1]['label'].replace('<K>', str(k_val_for_label)))

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save:
        y_part = '_'.join(y_axis_keys)
        output_filepath = f'plot_filtered_{x_axis_key}_{y_part}.png'
        output_filepath = output_filepath.replace(' ', '_')
        try:
            plt.savefig(output_filepath)
            print(f"Plot saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving plot to {output_filepath}: {e}", file=sys.stderr)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot latency vs. recall from filtered search results.")
    parser.add_argument("--stdout_file", type=Path, default='stdout_filtered.txt',
                        help="Path to the stdout_filtered.txt log file.")
    parser.add_argument("--stdout_file_naive", type=Path, default='stdout_filtered_naive.txt',
                        help="Path to the stdout_filtered_naive.txt log file.")
    parser.add_argument("--data_dir", type=Path, default='.',
                        help="Directory containing the  data files (filtered_naive_*, filtered_out_*).")
    parser.add_argument('-x', "--x_axis", type=str, default="latency",
                        help="Metric for the X-axis (currently only 'latency' is supported).")
    parser.add_argument('-y', "--y_axis", type=str, action='append',
                        help="Metric(s) for the Y-axis. Can be specified multiple times.")
    parser.add_argument("--y-label", type=str, default=None,
                        help="Metric for the Y-axis.")
    parser.add_argument("--k_val", type=int, default=K_DEFAULT,
                        help=f"The 'K' value for recall calculations (number of neighbors, default: {K_DEFAULT}).")
    parser.add_argument("--q_val", type=int, default=Q_DEFAULT,
                        help=f"The 'Q' value (number of queries, default: {Q_DEFAULT}).")
    parser.add_argument('--save', action='store_true', help='Save the output to a file.')

    args = parser.parse_args()

    if not args.stdout_file.is_file():
        print(f"Error: stdout_file '{args.stdout_file}' not found.", file=sys.stderr)
        sys.exit(1)
    if not args.data_dir.is_dir():
        print(f"Error: data_dir '{args.data_dir}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)

    print("Starting data processing...")
    prefilter_data = process_naive(args.stdout_file_naive)
    all_plot_data = process_data(args.stdout_file, args.data_dir, args.q_val, args.k_val)
    all_plot_data.update(prefilter_data) # no overlaps

    if not all_plot_data:
        print("No data was successfully processed. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Data processing complete. Generating plot...")
    generate_plot(all_plot_data, args.x_axis, args.y_axis or ['recall_1_k'], args.y_label, args.k_val, args.save)
    print("Script finished.")

if __name__ == "__main__":
    main()
