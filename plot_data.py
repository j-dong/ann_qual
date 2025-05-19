import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import argparse
import struct
import numpy as np

hnsw_color = 'C0'
vamana_color = 'C1'
pq_color = 'C2'

# --- Configuration for Recall-100@100 ---
GROUND_TRUTH_FILE_PATH_SIFT1M_100NN = 'G:\\vectors\\sift\\sift_groundtruth.ivecs'
_sift_groundtruth_data_100nn = None # For lazy loading

def load_sift_groundtruth_ivecs(filepath, num_queries=10000, expected_dims=100):
    """
    Loads SIFT ground truth data from an .ivecs file.
    Format: struct { uint32_t dims; uint32_t values[dims]; } repeated num_queries times.
    Returns a list of sets, where each set contains the ground truth neighbor IDs for a query.
    Returns None if file not found or format error.
    """
    ground_truth_sets = []
    try:
        with open(filepath, 'rb') as f:
            for _ in range(num_queries):
                dims_bytes = f.read(4)
                if not dims_bytes:
                    print(f"Error: Unexpected EOF while reading dims from {filepath}")
                    return None
                dims = struct.unpack('I', dims_bytes)[0]

                if dims != expected_dims:
                    print(f"Error: Expected dims {expected_dims}, got {dims} in {filepath}")
                    return None

                values_bytes = f.read(dims * 4)
                if len(values_bytes) != dims * 4:
                    print(f"Error: Unexpected EOF while reading values from {filepath}")
                    return None

                values = struct.unpack(f'{dims}I', values_bytes)
                ground_truth_sets.append(set(values))
        return ground_truth_sets
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading ground truth file {filepath}: {e}")
        return None

def calculate_recall_at_k(predicted_neighbors_filepath, ground_truth_sets, num_queries=10000, k=100):
    """
    Calculates recall@k by comparing predicted neighbors with ground truth.
    Predicted neighbors file is a flat list of uint32_t IDs (num_queries * k).
    Returns the average recall, or None if an error occurs.
    """
    try:
        # Read all predicted neighbor IDs at once
        predicted_ids = np.fromfile(predicted_neighbors_filepath, dtype=np.uint32, count=num_queries * k)
        if len(predicted_ids) != num_queries * k:
            print(f"Error: Expected {num_queries * k} IDs in {predicted_neighbors_filepath}, found {len(predicted_ids)}")
            return None

    except FileNotFoundError:
        print(f"Error: Predicted neighbors file not found: {predicted_neighbors_filepath}")
        return None
    except Exception as e:
        print(f"Error reading predicted neighbors file {predicted_neighbors_filepath}: {e}")
        return None

    total_recall = 0
    for i in range(num_queries):
        start_idx = i * k
        end_idx = start_idx + k

        predicted_set = set(predicted_ids[start_idx:end_idx])
        ground_truth_set = ground_truth_sets[i]

        intersection_size = len(predicted_set.intersection(ground_truth_set))
        total_recall += intersection_size / k

    return total_recall / num_queries if num_queries > 0 else 0

def get_recall_100_at_100_value_computer(data_point, base_dir='outs'):
    """
    Computes and returns the Recall-100@100 value for a given data_point.
    Handles lazy loading of ground truth.
    data_point is expected to have 'k_ann_output_file'.
    """
    global _sift_groundtruth_data_100nn

    if not data_point.get('k_ann_output_file'):
        # print(f"Warning: 'k_ann_output_file' missing for {data_point.get('filename', 'N/A')}, cannot compute Recall-100@100.")
        return None

    if _sift_groundtruth_data_100nn is None:
        print(f"Attempting to load ground truth from: {GROUND_TRUTH_FILE_PATH_SIFT1M_100NN}")
        _sift_groundtruth_data_100nn = load_sift_groundtruth_ivecs(GROUND_TRUTH_FILE_PATH_SIFT1M_100NN)
        if _sift_groundtruth_data_100nn is None:
            print("ERROR: Failed to load SIFT1M ground truth data. Recall-100@100 will not be available.")
            return None
        print("Ground truth data loaded successfully.")

    if _sift_groundtruth_data_100nn: # Check again in case loading failed
        k_ann_output_filename = data_point['k_ann_output_file']
        # k_ann_output_file might be a full path or relative.
        # Assuming it's a filename that should be in base_dir if not absolute.
        if not os.path.isabs(k_ann_output_filename):
             k_ann_full_path = os.path.join(base_dir, k_ann_output_filename)
        else:
            k_ann_full_path = k_ann_output_filename

        # print(f"Calculating Recall-100@100 for {k_ann_full_path}")
        return calculate_recall_at_k(k_ann_full_path, _sift_groundtruth_data_100nn)
    return None
# --- End of Recall-100@100 specific code ---

def get_latency(content):
    """Extracts latency from file content."""
    lat_match = re.search(r'avg query latency: ([\d\.]+) ms', content)
    if lat_match:
        return float(lat_match.group(1))
    return None

def get_recall_1_100(content): # This is the original recall, presumably R@100 where R=1
    """Extracts recall@100 from file content."""
    rec_match = re.search(r'recall@100: ([\d\.]+)', content)
    if rec_match:
        return float(rec_match.group(1))
    return None

def parse_filename(filename):
    """Extracts parameters from the filename."""
    params = {}
    # Ensure it's not one of the k-ANN output files like "out_hnsw_M128_efC128"
    if filename.startswith("out_"):
        return params # Not a primary log file

    if "hnsw" in filename :
        params['type'] = 'hnsw'
        match = re.search(r'M(\d+)', filename)
        if match:
            params['M'] = int(match.group(1))
    elif "vamana" in filename:
        params['type'] = 'vamana'
        match = re.search(r'R(\d+)', filename)
        if match:
            params['R'] = int(match.group(1))
    elif "pq" in filename:
        params['type'] = 'pq'
        match = re.search(r'K(\d+)_W(\d+)_B(\d+)_M(\d+)', filename)
        if match:
            params['K'] = int(match.group(1))
            params['W'] = int(match.group(2))
            params['B'] = int(match.group(3))
            params['M'] = int(match.group(4))
    return params

def parse_file_content(filepath):
    """
    Extracts data and k-ANN output filename from file content.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}")
        return None, None, None, None

    latency = get_latency(content)
    recall_1_at_100 = get_recall_1_100(content)

    k_ann_output_file = None
    k_ann_match = re.search(r"k-ANN output written to: (\S+)", content)
    if k_ann_match:
        k_ann_output_file = k_ann_match.group(1)

    # The function signature expects 4 return values now due to the initial code structure,
    # The 4th one (actual recall_100_100 value) will be computed later if needed.
    return latency, recall_1_at_100, k_ann_output_file, None


AXIS_METADATA = {
    'latency': {
        'title': 'Latency (ms)',
        'data_key': 'latency_value', # Changed to avoid conflict with params['latency']
    },
    'recall_1_100': { # This is the original recall parsed from logs (e.g. R@k for k=100, R=1)
        'title': 'Recall-1@100',
        'data_key': 'recall_1_100_value',
    },
    'recall_100_100': {
        'title': 'Recall-100@100',
        'data_key': 'recall_100_at_100_actual_value',
        'computer_func': get_recall_100_at_100_value_computer # Specific function to compute this
    }
}


def main():
    parser = argparse.ArgumentParser(description="Plot performance metrics.")
    parser.add_argument('--save', action='store_true', help='Save the output to a file.')
    parser.add_argument('-x', default='latency', choices=list(AXIS_METADATA.keys()),
                        help=f'Function name for X axis. Options: {", ".join(AXIS_METADATA.keys())}')
    parser.add_argument('-y', default='recall_1_100', choices=list(AXIS_METADATA.keys()),
                        help=f'Function name for Y axis. Options: {", ".join(AXIS_METADATA.keys())}')
    parser.add_argument('--hide-pq', action='store_true', help='Hide PQ data from the plot.')
    parser.add_argument('--outs-dir', default='outs', help='Directory containing the output log files.')

    args = parser.parse_args()

    if args.x not in AXIS_METADATA or args.y not in AXIS_METADATA:
        print("Error: Invalid function name for -x or -y argument.")
        return

    if not os.path.isdir(args.outs_dir):
        print(f"Error: Output directory '{args.outs_dir}' not found.")
        return

    filenames = [
        fn for fn in os.listdir(args.outs_dir) if fn.startswith('stdout')
    ]

    data = []
    for f_name in filenames:
        base_params = parse_filename(f_name)
        if not base_params:
            continue

        latency, recall_1_at_100, k_ann_output_file, _ = parse_file_content(os.path.join(args.outs_dir, f_name))

        current_data_point = {}
        current_data_point.update(base_params)
        current_data_point['filename'] = f_name
        current_data_point['k_ann_output_file'] = k_ann_output_file

        if latency is not None:
            current_data_point[AXIS_METADATA['latency']['data_key']] = latency
        if recall_1_at_100 is not None:
            current_data_point[AXIS_METADATA['recall_1_100']['data_key']] = recall_1_at_100

        # Compute values for axes if they have a computer_func
        for axis_arg_name in [args.x, args.y]:
            axis_meta = AXIS_METADATA[axis_arg_name]
            if 'computer_func' in axis_meta and axis_meta['data_key'] not in current_data_point:
                computed_value = axis_meta['computer_func'](current_data_point, base_dir=args.outs_dir)
                if computed_value is not None:
                    current_data_point[axis_meta['data_key']] = computed_value
                # else:
                    # print(f"Warning: Could not compute {axis_arg_name} for {f_name}")


        required_x_key = AXIS_METADATA[args.x]['data_key']
        required_y_key = AXIS_METADATA[args.y]['data_key']

        if required_x_key in current_data_point and required_y_key in current_data_point:
            data.append(current_data_point)
        # else:
        #     print(f"Warning: Missing required data for plot axes ({args.x} or {args.y}) for file {f_name}. Skipping.")


    # Separate data by type
    hnsw_data = sorted([d for d in data if d.get('type') == 'hnsw'], key=lambda x: x.get('M', float('inf')))
    vamana_data = sorted([d for d in data if d.get('type') == 'vamana'], key=lambda x: x.get('R', float('inf')))
    pq_data = [d for d in data if d.get('type') == 'pq']

    plt.figure(figsize=(12, 8))

    x_data_key = AXIS_METADATA[args.x]['data_key']
    y_data_key = AXIS_METADATA[args.y]['data_key']

    # Plot HNSW
    if hnsw_data:
        x_hnsw = [d[x_data_key] for d in hnsw_data if x_data_key in d]
        y_hnsw = [d[y_data_key] for d in hnsw_data if y_data_key in d]
        if x_hnsw and y_hnsw: # Ensure there's data to plot
            valid_hnsw_data = [d for d in hnsw_data if x_data_key in d and y_data_key in d]
            plt.plot([d[x_data_key] for d in valid_hnsw_data],
                     [d[y_data_key] for d in valid_hnsw_data],
                     marker='o', linestyle='-', color=hnsw_color, label='HNSW')
            for d in valid_hnsw_data:
                plt.text(d[x_data_key], d[y_data_key], f" M{d.get('M','')}", fontsize=8, va='bottom', ha='left')

    # Plot Vamana
    if vamana_data:
        x_vamana = [d[x_data_key] for d in vamana_data if x_data_key in d]
        y_vamana = [d[y_data_key] for d in vamana_data if y_data_key in d]
        if x_vamana and y_vamana: # Ensure there's data to plot
            valid_vamana_data = [d for d in vamana_data if x_data_key in d and y_data_key in d]
            plt.plot([d[x_data_key] for d in valid_vamana_data],
                     [d[y_data_key] for d in valid_vamana_data],
                     marker='s', linestyle='-', color=vamana_color, label='Vamana')
            for d in valid_vamana_data:
                plt.text(d[x_data_key], d[y_data_key], f" R{d.get('R','')}", fontsize=8, va='bottom', ha='left')


    legend_elements = []
    if any(d.get('type') == 'hnsw' and x_data_key in d and y_data_key in d for d in data):
         legend_elements.append(mlines.Line2D([0], [0], color=hnsw_color, marker='o', linestyle='-', label='HNSW'))
    if any(d.get('type') == 'vamana' and x_data_key in d and y_data_key in d for d in data):
        legend_elements.append(mlines.Line2D([0], [0], color=vamana_color, marker='s', linestyle='-', label='Vamana'))


    if not args.hide_pq:
        pq_k1024_valid = [d for d in pq_data if d.get('K') == 1024 and x_data_key in d and y_data_key in d]
        pq_k8192_valid = [d for d in pq_data if d.get('K') == 8192 and x_data_key in d and y_data_key in d]

        pq_k1024 = sorted(pq_k1024_valid, key=lambda x: (x.get('M', 0), x.get('W', 0)))
        pq_k8192 = sorted(pq_k8192_valid, key=lambda x: (x.get('M', 0), x.get('W', 0)))


        points_k1024 = {}
        for d in pq_k1024:
            points_k1024[(d['M'], d['W'])] = (d[x_data_key], d[y_data_key])
            plt.plot(d[x_data_key], d[y_data_key], marker='^', color=pq_color, markersize=8, linestyle='None')
            plt.text(d[x_data_key], d[y_data_key], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='top')

        points_k8192 = {}
        for d in pq_k8192:
            points_k8192[(d['M'], d['W'])] = (d[x_data_key], d[y_data_key])
            plt.plot(d[x_data_key], d[y_data_key], marker='x', color=pq_color, markersize=8, linestyle='None')
            plt.text(d[x_data_key], d[y_data_key], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='bottom')

        def connect_pq_quads(points_dict, line_color):
            m_values = sorted(list(set(m for m,w in points_dict.keys())))
            w_values = sorted(list(set(w for m,w in points_dict.keys())))

            if len(m_values) >= 2 and len(w_values) >= 2:
                for i in range(len(m_values) -1): # Iterate through M-pairs
                    m1 = m_values[i]
                    # This connection logic assumes M values are adjacent in the sorted list
                    # and form "quads" based on their sorted order.
                    # For arbitrary M pairings, a different grouping logic would be needed.
                    # The original script seemed to imply specific M-pairs like (8,32).
                    # This version will connect any two adjacent M's if they share W's.

                    # Try to connect M-pairs that share W values
                    for j in range(i + 1, len(m_values)):
                        m2 = m_values[j] # Consider all other M values as potential pairs

                        # M-edges (same W, different M) - dashed
                        for w_val in w_values:
                            p_m1w = points_dict.get((m1, w_val))
                            p_m2w = points_dict.get((m2, w_val))
                            if p_m1w and p_m2w:
                                plt.plot([p_m1w[0], p_m2w[0]], [p_m1w[1], p_m2w[1]], color=line_color, linestyle='--')

                # W-edges (same M, different W) - solid
                for m_val in m_values:
                     # Find points with this M value and different W values
                    current_m_points_w_sorted = sorted([(w, points_dict[(m_val,w)]) for w in w_values if (m_val,w) in points_dict], key=lambda item: item[0])

                    # Connect adjacent W values for the same M
                    for k_idx in range(len(current_m_points_w_sorted) - 1):
                        p_mw1 = current_m_points_w_sorted[k_idx][1]
                        p_mw2 = current_m_points_w_sorted[k_idx+1][1]
                        plt.plot([p_mw1[0], p_mw2[0]], [p_mw1[1], p_mw2[1]], color=line_color, linestyle='-')


        if points_k1024:
            connect_pq_quads(points_k1024, pq_color)
        if points_k8192:
            connect_pq_quads(points_k8192, pq_color)

        if pq_k1024_valid:
             legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='^', linestyle='None', label='PQ K=1024'))
        if pq_k8192_valid:
            legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='x', linestyle='None', label='PQ K=8192'))
        if pq_k1024_valid or pq_k8192_valid : # Add line style legends if any PQ data is plotted
            legend_elements.append(mlines.Line2D([0], [0], color=pq_color, linestyle='-', label='change w (same M)'))
            legend_elements.append(mlines.Line2D([0], [0], color=pq_color, linestyle='--', label='change M (same W)'))


    plt.xlabel(AXIS_METADATA[args.x]['title'])
    plt.ylabel(AXIS_METADATA[args.y]['title'])
    plt.title(f"{AXIS_METADATA[args.x]['title']} vs. {AXIS_METADATA[args.y]['title']}")

    if legend_elements: # Only show legend if there are items
        plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.tight_layout()

    if args.save:
        plot_filename = f"plot_{args.x}_{args.y}.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
