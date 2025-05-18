import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

hnsw_color = 'C0'
vamana_color = 'C1'
pq_color = 'C2'

def parse_filename(filename):
    """Extracts parameters from the filename."""
    params = {}
    if "hnsw" in filename:
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

def parse_file_content(filename):
    """
    Extracts latency and recall from file content.
    """
    latency = None
    recall = None
    content = open('outs/' + filename, 'r', encoding='utf-8').read()

    lat_match = re.search(r'avg query latency: ([\d\.]+) ms', content)
    if lat_match:
        latency = float(lat_match.group(1))

    rec_match = re.search(r'recall@100: ([\d\.]+)', content)
    if rec_match:
        recall = float(rec_match.group(1))
    return latency, recall

filenames = [
    fn for fn in os.listdir('outs') if fn.startswith('stdout')
]

data = []
for f_name in filenames:
    params = parse_filename(f_name)
    # In a real scenario, you would pass the full path to the file if needed.
    # For this example, f_name is just the key for placeholder data.
    latency, recall = parse_file_content(f_name)
    if latency is not None and recall is not None:
        params['latency'] = latency
        params['recall'] = recall
        params['filename'] = f_name
        data.append(params)
    else:
        print(f"Warning: Could not parse data for {f_name}")


# Separate data by type
hnsw_data = sorted([d for d in data if d['type'] == 'hnsw'], key=lambda x: x['M'])
vamana_data = sorted([d for d in data if d['type'] == 'vamana'], key=lambda x: x['R'])
pq_data = [d for d in data if d['type'] == 'pq']

# Plotting
plt.figure(figsize=(12, 8))

# Plot HNSW
if hnsw_data:
    x_hnsw = [d['latency'] for d in hnsw_data]
    y_hnsw = [d['recall'] for d in hnsw_data]
    plt.plot(x_hnsw, y_hnsw, marker='o', linestyle='-', color=hnsw_color, label='HNSW')
    for d in hnsw_data:
        plt.text(d['latency'], d['recall'], f" M{d['M']}", fontsize=8, va='bottom', ha='left')

# Plot Vamana
if vamana_data:
    x_vamana = [d['latency'] for d in vamana_data]
    y_vamana = [d['recall'] for d in vamana_data]
    plt.plot(x_vamana, y_vamana, marker='s', linestyle='-', color=vamana_color, label='Vamana')
    for d in vamana_data:
        plt.text(d['latency'], d['recall'], f" R{d['R']}", fontsize=8, va='bottom', ha='left')

# Plot PQ
pq_k1024 = sorted([d for d in pq_data if d.get('K') == 1024], key=lambda x: (x.get('M', 0), x.get('W', 0)))
pq_k8192 = sorted([d for d in pq_data if d.get('K') == 8192], key=lambda x: (x.get('M', 0), x.get('W', 0)))


# K=1024
points_k1024 = {} # Store points for connecting lines: {(M,W): (lat, rec)}
for d in pq_k1024:
    points_k1024[(d['M'], d['W'])] = (d['latency'], d['recall'])
    plt.plot(d['latency'], d['recall'], marker='^', color=pq_color, markersize=8, linestyle='None')
    plt.text(d['latency'], d['recall'], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='top')

# K=8192
points_k8192 = {}
for d in pq_k8192:
    points_k8192[(d['M'], d['W'])] = (d['latency'], d['recall'])
    plt.plot(d['latency'], d['recall'], marker='x', color=pq_color, markersize=8, linestyle='None')
    plt.text(d['latency'], d['recall'], f" K{d['K']}\nW{d['W']}\nM{d['M']}", fontsize=7, ha='center', va='bottom')


# Connect PQ series as quadrilaterals
def connect_pq_quads(points_dict, line_color):
    m_values = sorted(list(set(m for m,w in points_dict.keys())))
    w_values = sorted(list(set(w for m,w in points_dict.keys())))

    # We expect two M values and two W values for a quadrilateral
    if len(m_values) >= 2 and len(w_values) >= 2:
        # Iterate through M values to form M-pairs for quadrilaterals
        for i in range(len(m_values) -1):
            m1 = m_values[i]
            m2 = m_values[i+1] # Potential second M for a quad

            # Check for specific M values from the dataset (8 and 32)
            # This logic assumes specific M values for quadrilaterals.
            # A more general approach might be needed for arbitrary M values forming quads.
            # For now, let's assume we are looking for M=8 and M=32 quads.
            if not ((m1 == 8 and m2 == 32) or (m1 == 32 and m2 == 8)): # Ensure we have the M-pair for a quad
                 # Allow any two M values to form edges if W values match
                 pass # Continue to allow any two M values for edges if W values match


            # W-edges (same M, different W) - solid
            for m_val in [m1, m2]: # Iterate through both M values
                 # Find points with this M value and different W values
                current_m_points_w = sorted([(w, points_dict[(m_val,w)]) for w in w_values if (m_val,w) in points_dict])
                if len(current_m_points_w) == 2: # Expecting two W values for each M
                    p_mw1 = current_m_points_w[0][1]
                    p_mw2 = current_m_points_w[1][1]
                    plt.plot([p_mw1[0], p_mw2[0]], [p_mw1[1], p_mw2[1]], color=line_color, linestyle='-')


            # M-edges (same W, different M) - dashed
            for w_val in w_values: # Iterate through W values
                # Find points with this W value and the two M values (m1, m2)
                p_m1w = points_dict.get((m1, w_val))
                p_m2w = points_dict.get((m2, w_val))
                if p_m1w and p_m2w:
                     plt.plot([p_m1w[0], p_m2w[0]], [p_m1w[1], p_m2w[1]], color=line_color, linestyle='--')
    else:
        print(f"Not enough M ({len(m_values)}) or W ({len(w_values)}) values to form quadrilaterals as specified.")


if points_k1024:
    connect_pq_quads(points_k1024, pq_color)
if points_k8192:
    connect_pq_quads(points_k8192, pq_color)


# Create legend handles for PQ series
legend_elements = [
    mlines.Line2D([0], [0], color=hnsw_color, marker='o', linestyle='-', label='HNSW'),
    mlines.Line2D([0], [0], color=vamana_color, marker='s', linestyle='-', label='Vamana'),
    mlines.Line2D([0], [0], color=pq_color, marker='^', linestyle='None', label='PQ K=1024'),
    mlines.Line2D([0], [0], color=pq_color, marker='x', linestyle='None', label='PQ K=8192'),
    mlines.Line2D([0], [0], color=pq_color, linestyle='-', label='change w'),
    mlines.Line2D([0], [0], color=pq_color, linestyle='--', label='change m')
]


plt.xlabel("Average Query Latency (ms)")
plt.ylabel("Recall@100")
plt.title("Latency vs. Recall")
plt.legend(handles=legend_elements)
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
plot_filename = "latency_recall_plot.png"
# plt.savefig(plot_filename)
plt.show() # Comment out or remove for non-interactive environments like automated scripts

# print(f"Plot saved as {plot_filename}")
