import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import argparse
import struct
import numpy as np
import matplotlib.axes
import typing

hnsw_color = 'C0'
vamana_color = 'C1'
pq_color = 'C2'

N = 1000000

# --- Configuration for Recall-100@100 ---
GROUND_TRUTH_FILE_PATH_SIFT1M_100NN = 'G:\\vectors\\sift\\sift_groundtruth.ivecs'
_sift_groundtruth_data_100nn = None # For lazy loading
_graph_metadata_cache = {}

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
            num_queries = len(predicted_ids) // k

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

def get_max_degree_value_computer(data_point, **kwargs):
    """
    Computes the maximum degree based on algorithm type and parameters.
    HNSW: 2 * M
    Vamana: R
    """
    algo_type = data_point.get('type')
    if algo_type == 'hnsw':
        m_val = data_point.get('M')
        if m_val is not None:
            return 2 * m_val
    elif algo_type == 'vamana':
        r_val = data_point.get('R')
        if r_val is not None:
            return r_val
    # PQ or other types do not have a defined max degree in this context
    return None

def load_graph_metadata(graph_filepath):
    """
    Loads graph metadata from a binary file (V x 4 matrix of uint32_t).
    Caches the loaded data to avoid redundant reads.
    Each record: id, level, degree, distance.
    Returns a NumPy array (V, 4) or None if an error occurs.
    """
    global _graph_metadata_cache
    if graph_filepath in _graph_metadata_cache:
        # Return cached data, even if it was a failure (None)
        return _graph_metadata_cache[graph_filepath]

    try:
        data = np.fromfile(graph_filepath, dtype=np.uint32)
        if data.size == 0:
            # print(f"Warning: Graph metadata file is empty: {graph_filepath}") # Less verbose
            _graph_metadata_cache[graph_filepath] = None
            return None
        if data.size % 4 != 0:
            print(f"Warning: Graph metadata file size {data.size} is not a multiple of 4 fields for {graph_filepath}")
            _graph_metadata_cache[graph_filepath] = None
            return None

        graph_matrix = data.reshape(-1, 4)
        _graph_metadata_cache[graph_filepath] = graph_matrix
        # print(f"Successfully loaded graph metadata for {graph_filepath}, shape {graph_matrix.shape}")
        return graph_matrix
    except FileNotFoundError:
        # print(f"Warning: Graph metadata file not found: {graph_filepath}") # Less verbose
        _graph_metadata_cache[graph_filepath] = None
        return None
    except Exception as e:
        print(f"Error loading graph metadata from {graph_filepath}: {e}")
        _graph_metadata_cache[graph_filepath] = None
        return None

def get_avg_vertex_degree_computer(data_point, base_dir='outs'):
    """
    Computes the average vertex degree from graph metadata.
    """
    if not data_point.get('k_ann_output_file'):
        return None

    graph_filename = 'graph_' + data_point['k_ann_output_file']
    graph_filepath = os.path.join(base_dir, graph_filename)

    graph_matrix = load_graph_metadata(graph_filepath)
    if graph_matrix is None or graph_matrix.shape[0] == 0:
        # print(f"Could not load or empty graph data for {graph_filepath} to compute avg distance.")
        return None

    degrees = graph_matrix[:N, 2].astype(np.uint32)

    return np.mean(degrees)

def get_avg_vertex_distance_computer(data_point, base_dir='outs'):
    """
    Computes the average vertex distance from graph metadata.
    Distance is stored as its one's complement if not found (MSB=1).
    "Not-found" distances (after undoing complement) are included in the average.
    """
    if not data_point.get('k_ann_output_file'):
        return None

    graph_filename = 'graph_' + data_point['k_ann_output_file']
    graph_filepath = os.path.join(base_dir, graph_filename)

    graph_matrix = load_graph_metadata(graph_filepath)
    if graph_matrix is None or graph_matrix.shape[0] == 0:
        # print(f"Could not load or empty graph data for {graph_filepath} to compute avg distance.")
        return None

    stored_distances = graph_matrix[:N, 3].astype(np.uint32) # 4th column is distance

    # If MSB is 0, distance is as-is.
    # If MSB is 1, it's a "not found" marker, stored as ~original_marker_value.
    # We use ~stored_value to get that original_marker_value for the average.
    processed_distances = np.where(
        (stored_distances & 0x80000000) == 0,  # Condition: MSB is 0 (found)
        stored_distances,                       # Value if true
        ~stored_distances                       # Value if false (apply NOT to complemented "not found" marker)
    ).astype(np.uint32) # Ensure it remains uint32 after potential bitwise not on mixed types

    return np.mean(processed_distances)

def get_vertex_recall_computer(data_point, base_dir='outs'):
    """
    Computes vertex recall: fraction of vertices where distance is found.
    Distance is considered "found" if its MSB is 0 in the stored graph metadata.
    """
    if not data_point.get('k_ann_output_file'):
        return None

    graph_filename = 'graph_' + data_point['k_ann_output_file']
    graph_filepath = os.path.join(base_dir, graph_filename)

    graph_matrix = load_graph_metadata(graph_filepath)
    if graph_matrix is None or graph_matrix.shape[0] == 0:
        # print(f"Could not load or empty graph data for {graph_filepath} to compute vertex recall.")
        return None

    stored_distances = graph_matrix[:N, 3].astype(np.uint32) # 4th column
    num_vertices = stored_distances.shape[0]

    if num_vertices == 0:
        return 0.0 # Or None, depending on desired behavior for empty graphs

    # Count vertices where distance is "found" (MSB is 0)
    num_found = np.sum((stored_distances & 0x80000000) == 0)

    return num_found / num_vertices

# --- etc ---

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

def get_avg_vertices_explored(content):
    """Extracts average number of vertices explored from file content."""
    match = re.search(r"\[STATS\] avg num vertices explored: ([\d\.]+)", content)
    if match:
        return float(match.group(1))
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
    avg_vertices = get_avg_vertices_explored(content)

    k_ann_output_file = None
    k_ann_match = re.search(r"k-ANN output written to: (\S+)", content)
    if k_ann_match:
        k_ann_output_file = k_ann_match.group(1)

    return latency, recall_1_at_100, k_ann_output_file, avg_vertices


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
    },
    'degree': {
        'title': 'Maximum Degree',
        'data_key': 'max_degree_value',
        'computer_func': get_max_degree_value_computer
    },
    'explored': {
        'title': 'Avg. Vertices Explored',
        'data_key': 'avg_vertices_explored_value'
        # This will be directly populated from parse_file_content output
    },
    'avg_vertex_distance': {
        'title': 'Avg. Vertex Distance',
        'data_key': 'avg_vertex_distance_value',
        'computer_func': get_avg_vertex_distance_computer
    },
    'avg_degree': {
        'title': 'Avg. Degree',
        'data_key': 'avg_vertex_degree',
        'computer_func': get_avg_vertex_degree_computer
    },
    'vertex_recall': {
        'title': 'Vertex Recall',
        'data_key': 'vertex_recall_value',
        'computer_func': get_vertex_recall_computer
    },
}

def connect_pq_quads(points_dict, line_color, ax): # Added ax argument
    m_values = sorted(list(set(m for m,w in points_dict.keys())))
    w_values = sorted(list(set(w for m,w in points_dict.keys())))

    if len(m_values) >= 2 and len(w_values) >= 2:
        # ... (rest of the logic, but use ax.plot instead of plt.plot)
        # Example change:
        # plt.plot([p_mw1[0], p_mw2[0]], [p_mw1[1], p_mw2[1]], color=line_color, linestyle='-')
        # becomes:
        # ax.plot([p_mw1[0], p_mw2[0]], [p_mw1[1], p_mw2[1]], color=line_color, linestyle='-')
        # This change needs to be applied to all plt.plot calls within connect_pq_quads
        # For brevity, I'm not rewriting the whole function here but indicating the change pattern.
        # Search and replace plt.plot with ax.plot inside connect_pq_quads.

        # W-edges (same M, different W) - solid
        for m_val in m_values:
            current_m_points_w_sorted = sorted([(w, points_dict[(m_val,w)]) for w in w_values if (m_val,w) in points_dict], key=lambda item: item[0])
            for k_idx in range(len(current_m_points_w_sorted) - 1):
                p_mw1 = current_m_points_w_sorted[k_idx][1]
                p_mw2 = current_m_points_w_sorted[k_idx+1][1]
                ax.plot([p_mw1[0], p_mw2[0]], [p_mw1[1], p_mw2[1]], color=line_color, linestyle='-')

        # M-edges (same W, different M) - dashed
        # Iterate through M value pairs
        for i in range(len(m_values)):
            for j in range(i + 1, len(m_values)):
                m1 = m_values[i]
                m2 = m_values[j]
                for w_val in w_values:
                    p_m1w = points_dict.get((m1, w_val))
                    p_m2w = points_dict.get((m2, w_val))
                    if p_m1w and p_m2w:
                        ax.plot([p_m1w[0], p_m2w[0]], [p_m1w[1], p_m2w[1]], color=line_color, linestyle='--')

    # else:
        # print(f"Not enough M ({len(m_values)}) or W ({len(w_values)}) values to form quadrilaterals for {points_dict.keys()} on axis {ax}")


def main():
    parser = argparse.ArgumentParser(description="Plot performance metrics.")
    parser.add_argument('--save', action='store_true', help='Save the output to a file.')
    parser.add_argument('-x', default='latency', choices=list(AXIS_METADATA.keys()),
                        help=f'Function name for X axis. Options: {", ".join(AXIS_METADATA.keys())}')
    parser.add_argument('-y', default='recall_1_100', choices=list(AXIS_METADATA.keys()),
                        help=f'Function name for Y axis. Options: {", ".join(AXIS_METADATA.keys())}')
    parser.add_argument('-y2', default=None, choices=list(AXIS_METADATA.keys()) + [None], # Allow None
                        help='Optional: Function name for the second Y-axis. Clears PQ quads for Y1 if used.')
    parser.add_argument('--hide-pq', action='store_true', help='Hide PQ data from the plot.')
    parser.add_argument('--outs-dir', default='outs', help='Directory containing the output log files.')

    args = parser.parse_args()

    y1_axis_color = 'dimgray' # Color for Y1 axis label and ticks
    y2_axis_color = 'darkcyan' # Color for Y2 axis label and ticks

    if args.x not in AXIS_METADATA or args.y not in AXIS_METADATA:
        print("Error: Invalid function name for -x or -y argument.")
        return
    if args.y2 and args.y2 not in AXIS_METADATA:
        print("Error: Invalid function name for -y2 argument.")
        return
    if args.y == args.y2 and args.y2 is not None:
        print("Error: -y and -y2 cannot be the same metric.")
        return

    if not os.path.isdir(args.outs_dir):
        print(f"Error: Output directory '{args.outs_dir}' not found.")
        return

    filenames = [
        fn for fn in os.listdir(args.outs_dir) if fn.startswith('stdout')
    ]

    data = []

    active_metric_names = {args.x, args.y}
    if args.y2:
        active_metric_names.add(args.y2)

    for f_name in filenames:
        base_params = parse_filename(f_name)
        if not base_params:
            continue

        latency, recall_1_at_100, k_ann_output_file, avg_vertices_val = parse_file_content(os.path.join(args.outs_dir, f_name))

        current_data_point = {}
        current_data_point.update(base_params)
        current_data_point['filename'] = f_name
        current_data_point['k_ann_output_file'] = k_ann_output_file

        if latency is not None:
            current_data_point[AXIS_METADATA['latency']['data_key']] = latency
        if recall_1_at_100 is not None:
            current_data_point[AXIS_METADATA['recall_1_100']['data_key']] = recall_1_at_100
        if avg_vertices_val is not None:
            current_data_point[AXIS_METADATA['explored']['data_key']] = avg_vertices_val

        # Compute values for axes if they have a computer_func
        for metric_name in active_metric_names:
            if metric_name is None: continue # Should not happen with set logic but good check
            axis_meta = AXIS_METADATA[metric_name]
            # Ensure data_key exists before checking if it's already populated
            if 'computer_func' in axis_meta and axis_meta.get('data_key') not in current_data_point :
                computed_value = axis_meta['computer_func'](current_data_point, base_dir=args.outs_dir)
                if computed_value is not None:
                    current_data_point[axis_meta['data_key']] = computed_value


        required_x_key = AXIS_METADATA[args.x]['data_key']
        required_y1_key = AXIS_METADATA[args.y]['data_key']

        if required_x_key in current_data_point and required_y1_key in current_data_point:
            # If y2 is specified, it's desirable but not strictly required for the point to be added to master list.
            # Plotting functions will handle missing y2 data for specific series.
            data.append(current_data_point)
        # else:
            # print(f"Warning: Missing X or Y1 data for {f_name}. Point not added.")


    # Separate data by type
    hnsw_data = sorted([d for d in data if d.get('type') == 'hnsw'], key=lambda x: x.get('M', float('inf')))
    vamana_data = sorted([d for d in data if d.get('type') == 'vamana'], key=lambda x: x.get('R', float('inf')))
    pq_data = [d for d in data if d.get('type') == 'pq']

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = None

    if args.y2:
        ax2: typing.Optional[matplotlib.axes.Axes] = ax1.twinx() # type:ignore

    x_data_key = AXIS_METADATA[args.x]['data_key']
    y1_data_key = AXIS_METADATA[args.y]['data_key']
    y2_data_key = AXIS_METADATA[args.y2]['data_key'] if args.y2 else None

    legend_elements = []

    # --- HNSW Plotting (Modified) ---
    hnsw_data_points = sorted([d for d in data if d.get('type') == 'hnsw'], key=lambda x: x.get('M', float('inf')))
    if hnsw_data_points:
        valid_hnsw_y1 = [d for d in hnsw_data_points if x_data_key in d and y1_data_key in d]
        if valid_hnsw_y1:
            x_coords = [d[x_data_key] for d in valid_hnsw_y1]
            y1_coords = [d[y1_data_key] for d in valid_hnsw_y1]
            label = 'HNSW'
            if args.y2: label = f'HNSW ({AXIS_METADATA[args.y]["title"]})'
            line1, = ax1.plot(x_coords, y1_coords, marker='o', linestyle='-', color=hnsw_color, label=label)
            if not args.y2: # Add legend element only if not adding compound Y2 legend later
                 legend_elements.append(line1)
            for d in valid_hnsw_y1:
                ax1.text(d[x_data_key], d[y1_data_key], f" M{d.get('M','')}", fontsize=8, va='bottom', ha='left')

            if ax2 and y2_data_key:
                valid_hnsw_y2 = [d for d in hnsw_data_points if x_data_key in d and y2_data_key in d]
                if valid_hnsw_y2:
                    x_coords_y2 = [d[x_data_key] for d in valid_hnsw_y2]
                    y2_coords = [d[y2_data_key] for d in valid_hnsw_y2]
                    line2, = ax2.plot(x_coords_y2, y2_coords, marker='X', linestyle='--', color=hnsw_color, label=f'HNSW ({AXIS_METADATA[args.y2]["title"]})')
                    # If Y2 is present, a combined legend entry might be better.
                    # For now, let legend handle both distinct entries if line1 was also added.
                    # Or, create a combined legend element if line1 was for the same base series.
                    # For simplicity, we add both if y2 is active:
                    if valid_hnsw_y1 : legend_elements.append(line1) # Re-add Y1 line for clarity with Y2
                    legend_elements.append(line2)


    # --- Vamana Plotting (Modified similarly) ---
    vamana_data_points = sorted([d for d in data if d.get('type') == 'vamana'], key=lambda x: x.get('R', float('inf')))
    if vamana_data_points:
        valid_vamana_y1 = [d for d in vamana_data_points if x_data_key in d and y1_data_key in d]
        if valid_vamana_y1:
            x_coords = [d[x_data_key] for d in valid_vamana_y1]
            y1_coords = [d[y1_data_key] for d in valid_vamana_y1]
            line1_v, = ax1.plot(x_coords, y1_coords, marker='s', linestyle='-', color=vamana_color, label=f'Vamana ({AXIS_METADATA[args.y]["title"]})')
            if not args.y2: legend_elements.append(line1_v)
            for d in valid_vamana_y1:
                ax1.text(d[x_data_key], d[y1_data_key], f" R{d.get('R','')}", fontsize=8, va='bottom', ha='left')

            if ax2 and y2_data_key:
                valid_vamana_y2 = [d for d in vamana_data_points if x_data_key in d and y2_data_key in d]
                if valid_vamana_y2:
                    x_coords_y2 = [d[x_data_key] for d in valid_vamana_y2]
                    y2_coords = [d[y2_data_key] for d in valid_vamana_y2]
                    line2_v, = ax2.plot(x_coords_y2, y2_coords, marker='P', linestyle='--', color=vamana_color, label=f'Vamana ({AXIS_METADATA[args.y2]["title"]})')
                    if valid_vamana_y1: legend_elements.append(line1_v)
                    legend_elements.append(line2_v)


    # --- PQ Plotting (Modified - Y2 plots points only, Y1 quads conditional) ---
    if not args.hide_pq:
        pq_data_points = [d for d in data if d.get('type') == 'pq']

        # PQ K=1024
        pq_k1024_y1 = sorted([d for d in pq_data_points if d.get('K') == 1024 and x_data_key in d and y1_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
        points_k1024_y1 = {}
        if pq_k1024_y1:
            for d in pq_k1024_y1:
                points_k1024_y1[(d['M'], d['W'])] = (d[x_data_key], d[y1_data_key])
                ax1.plot(d[x_data_key], d[y1_data_key], marker='^', color=pq_color, markersize=8, linestyle='None')
                # Text only for Y1 to avoid overlap
                ax1.text(d[x_data_key], d[y1_data_key], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='top')
            if not ax2 : # Only connect quads if no Y2 axis, or define behavior
                connect_pq_quads(points_k1024_y1, pq_color, ax1) # Pass ax1
            legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='^', linestyle='None', label=f'PQ K1024 ({AXIS_METADATA[args.y]["title"]})'))


        if ax2 and y2_data_key:
            pq_k1024_y2 = sorted([d for d in pq_data_points if d.get('K') == 1024 and x_data_key in d and y2_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
            if pq_k1024_y2:
                for d in pq_k1024_y2:
                    ax2.plot(d[x_data_key], d[y2_data_key], marker='v', color=pq_color, markersize=7, linestyle='None') 
                legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='v', linestyle='None', label=f'PQ K1024 ({AXIS_METADATA[args.y2]["title"]})'))

        # PQ K=8192 (similar modifications)
        pq_k8192_y1 = sorted([d for d in pq_data_points if d.get('K') == 8192 and x_data_key in d and y1_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
        points_k8192_y1 = {}
        if pq_k8192_y1:
            for d in pq_k8192_y1:
                points_k8192_y1[(d['M'], d['W'])] = (d[x_data_key], d[y1_data_key])
                ax1.plot(d[x_data_key], d[y1_data_key], marker='x', color=pq_color, markersize=8, linestyle='None')
                ax1.text(d[x_data_key], d[y1_data_key], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='bottom')
            if not ax2:
                connect_pq_quads(points_k8192_y1, pq_color, ax1) # Pass ax1
            legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='x', linestyle='None', label=f'PQ K8192 ({AXIS_METADATA[args.y]["title"]})'))

        if ax2 and y2_data_key:
            pq_k8192_y2 = sorted([d for d in pq_data_points if d.get('K') == 8192 and x_data_key in d and y2_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
            if pq_k8192_y2:
                for d in pq_k8192_y2:
                    ax2.plot(d[x_data_key], d[y2_data_key], marker='+', color=pq_color, markersize=7, linestyle='None')
                legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='+', linestyle='None', label=f'PQ K8192 ({AXIS_METADATA[args.y2]["title"]})'))

        # Add PQ line style legends only if PQ data was plotted for Y1 and quads were relevant
        if (pq_k1024_y1 or pq_k8192_y1) and not ax2 : # Or if Y1 quads are always drawn
            legend_elements.append(mlines.Line2D([0], [0], color=pq_color, linestyle='-', label='PQ change w (Y1)'))
            legend_elements.append(mlines.Line2D([0], [0], color=pq_color, linestyle='--', label='PQ change M (Y1)'))

        if not args.hide_pq:
            pq_data_points = [d for d in data if d.get('type') == 'pq']

            # PQ K=1024
            pq_k1024_y1 = sorted([d for d in pq_data_points if d.get('K') == 1024 and x_data_key in d and y1_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
            points_k1024_y1 = {}
            if pq_k1024_y1:
                for d in pq_k1024_y1:
                    points_k1024_y1[(d['M'], d['W'])] = (d[x_data_key], d[y1_data_key])
                    ax1.plot(d[x_data_key], d[y1_data_key], marker='^', color=pq_color, markersize=8, linestyle='None')
                    # Text only for Y1 to avoid overlap
                    ax1.text(d[x_data_key], d[y1_data_key], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='top')
                if not ax2 : # Only connect quads if no Y2 axis, or define behavior
                    connect_pq_quads(points_k1024_y1, pq_color, ax1) # Pass ax1
                legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='^', linestyle='None', label=f'PQ K1024 ({AXIS_METADATA[args.y]["title"]})'))


            if ax2 and y2_data_key:
                pq_k1024_y2 = sorted([d for d in pq_data_points if d.get('K') == 1024 and x_data_key in d and y2_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
                if pq_k1024_y2:
                    for d in pq_k1024_y2:
                        ax2.plot(d[x_data_key], d[y2_data_key], marker='v', color=pq_color, markersize=7, linestyle='None') 
                    legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='v', linestyle='None', label=f'PQ K1024 ({AXIS_METADATA[args.y2]["title"]})'))

            # PQ K=8192 (similar modifications)
            pq_k8192_y1 = sorted([d for d in pq_data_points if d.get('K') == 8192 and x_data_key in d and y1_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
            points_k8192_y1 = {}
            if pq_k8192_y1:
                for d in pq_k8192_y1:
                    points_k8192_y1[(d['M'], d['W'])] = (d[x_data_key], d[y1_data_key])
                    ax1.plot(d[x_data_key], d[y1_data_key], marker='x', color=pq_color, markersize=8, linestyle='None')
                    ax1.text(d[x_data_key], d[y1_data_key], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='bottom')
                if not ax2:
                    connect_pq_quads(points_k8192_y1, pq_color, ax1) # Pass ax1
                legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='x', linestyle='None', label=f'PQ K8192 ({AXIS_METADATA[args.y]["title"]})'))

            if ax2 and y2_data_key:
                pq_k8192_y2 = sorted([d for d in pq_data_points if d.get('K') == 8192 and x_data_key in d and y2_data_key in d], key=lambda x: (x.get('M', 0), x.get('W', 0)))
                if pq_k8192_y2:
                    for d in pq_k8192_y2:
                        ax2.plot(d[x_data_key], d[y2_data_key], marker='+', color=pq_color, markersize=7, linestyle='None')
                    legend_elements.append(mlines.Line2D([0], [0], color=pq_color, marker='+', linestyle='None', label=f'PQ K8192 ({AXIS_METADATA[args.y2]["title"]})'))

            # Add PQ line style legends only if PQ data was plotted for Y1 and quads were relevant
            if (pq_k1024_y1 or pq_k8192_y1) and not ax2 : # Or if Y1 quads are always drawn
                legend_elements.append(mlines.Line2D([0], [0], color=pq_color, linestyle='-', label='PQ change w (Y1)'))
                legend_elements.append(mlines.Line2D([0], [0], color=pq_color, linestyle='--', label='PQ change M (Y1)'))

    if args.x == 'degree' and args.y == 'avg_degree':
        xs = [8, 16, 32, 64, 128, 256]
        ax1.plot(xs, xs, marker='None', color='lightgray')

    ax1.set_xlabel(AXIS_METADATA[args.x]['title'])
    ax1.set_ylabel(AXIS_METADATA[args.y]['title'], color=y1_axis_color)
    ax1.tick_params(axis='y', labelcolor=y1_axis_color)

    if ax2:
        ax2.set_ylabel(AXIS_METADATA[args.y2]['title'], color=y2_axis_color)
        ax2.tick_params(axis='y', labelcolor=y2_axis_color)
        fig.tight_layout() # Adjust layout to make room for the second y-axis
    else:
        plt.tight_layout()


    plot_title = f"{AXIS_METADATA[args.x]['title']} vs. {AXIS_METADATA[args.y]['title']}"
    if args.y2:
        plot_title += f" & {AXIS_METADATA[args.y2]['title']}"
    plt.title(plot_title)

    if legend_elements:
        # Position legend to avoid overlap, may need adjustment
        ax1.legend(handles=legend_elements, loc='best')
        # For very complex legends, fig.legend() or manual placement might be better.

    ax1.grid(True, linestyle=':', alpha=0.7) # Grid for primary axis

    fig.tight_layout()

    if args.save:
        plot_filename = f"plot_{args.x}_{args.y}"
        if args.y2:
            plot_filename += f"_vs_{args.y2}"
        plot_filename += ".png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
