import os
import struct
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Regex to capture FILE_NAME and PROB.
# It handles cases like "stats_filtered_out_tx_goodput_0.0"
# and also "stats_filtered_out_tx_goodput_0.0_ext_event_stats"
# by making the later part optional and non-capturing.
FILE_PATTERN = r"stats_filtered_out_(.*?)_([0-9.]+)(?:_ext_event_stats)?$"
MARKERS = ['o', 's', '^', 'P', 'D', '*', 'X', 'v', '<', '>'] # Markers for different FILE_NAMEs

# --- Helper Functions ---

def parse_filename(filename):
    """
    Parses the filename to extract FILE_NAME and PROB.
    Returns (file_name_base, prob) or None if no match.
    """
    match = re.match(FILE_PATTERN, os.path.basename(filename))
    if match:
        file_name_base = match.group(1)
        try:
            prob = float(match.group(2))
            return file_name_base, prob
        except ValueError:
            print(f"Warning: Could not parse PROB as float from filename: {filename}")
            return None
    return None

def read_stats_file(filepath):
    """
    Reads a single stats file according to the specified format.
    Returns a dictionary with parsed data or None on error.
    """
    try:
        with open(filepath, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0] # <I for little-endian uint32_t
            data_offset = struct.unpack('<I', f.read(4))[0]
            num_fields = struct.unpack('<I', f.read(4))[0]

            if num_fields == 0:
                print(f"Warning: num_fields is 0 in {filepath}. Skipping.")
                return None

            # Read field_sizes
            # Assuming each element in field_sizes is uint32_t (4 bytes)
            # and indicates the size of the corresponding field type.
            # Since "uint32_t fields[N][num_fields]" follows, each actual data field is 4 bytes.
            # So, each value in field_sizes should ideally be 4.
            field_sizes_values = []
            for _ in range(num_fields):
                field_sizes_values.append(struct.unpack('<I', f.read(4))[0])

            if not all(fs == 4 for fs in field_sizes_values):
                print(f"Warning: Not all field_sizes are 4 in {filepath}. Values: {field_sizes_values}. "
                      f"Proceeding based on 'uint32_t fields' (4 bytes per field value).")

            # Read field_descriptions
            current_pos = 4 + 4 + 4 + (4 * num_fields)
            desc_bytes_len = data_offset - current_pos

            if desc_bytes_len < 0:
                print(f"Error: data_offset ({data_offset}) is invalid in {filepath}. "
                      f"Calculated header size up to field_descriptions: {current_pos}. Skipping.")
                return None

            field_descriptions_bytes = f.read(desc_bytes_len)
            # Decode, remove null terminators, then split by newline.
            # Handles multiple null bytes at the end before splitting.
            try:
                # Find the first null byte to terminate the main string, then decode
                null_term_idx = field_descriptions_bytes.find(b'\0')
                if null_term_idx != -1:
                    active_desc_bytes = field_descriptions_bytes[:null_term_idx]
                else: # No null terminator found within the allocated space
                    active_desc_bytes = field_descriptions_bytes.rstrip(b'\0') # Strip any trailing just in case

                field_descriptions_str = active_desc_bytes.decode('utf-8')
                field_descriptions_list = [desc.strip() for desc in field_descriptions_str.split('\n') if desc.strip()]
            except UnicodeDecodeError:
                print(f"Error: Could not decode field_descriptions as UTF-8 in {filepath}. Skipping.")
                return None

            if len(field_descriptions_list) != num_fields:
                actual_num_desc = len(field_descriptions_list)
                print(f"Warning: Number of parsed field descriptions ({actual_num_desc}) "
                      f"does not match num_fields ({num_fields}) in {filepath}. "
                      f"Descriptions found: {field_descriptions_list}. Using parsed descriptions.")
                # Potentially adjust num_fields if descriptions are the source of truth,
                # or handle carefully during plotting. For now, we'll use the parsed descriptions.
                # If this happens, plotting might use fewer field names than num_fields suggests.

            # Seek to data_offset
            f.seek(data_offset)

            # Read data fields
            all_field_data = []
            bytes_per_row = num_fields * 4 # Each field is uint32_t (4 bytes)
            while True:
                row_bytes = f.read(bytes_per_row)
                if not row_bytes: # End of file
                    break
                if len(row_bytes) < bytes_per_row:
                    print(f"Warning: Incomplete data row at end of file {filepath}. "
                          f"Expected {bytes_per_row} bytes, got {len(row_bytes)}. Skipping partial row.")
                    break

                # Unpack based on the original num_fields read from header
                row_data = struct.unpack(f'<{num_fields}I', row_bytes)
                all_field_data.append(list(row_data))

            if not all_field_data:
                print(f"Note: No data rows found in {filepath} at offset {data_offset}.")
                # We can still proceed if header was fine, field_data will be empty.

            return {
                "magic": magic,
                "data_offset": data_offset,
                "num_fields_header": num_fields, # num_fields from header
                "field_sizes": field_sizes_values,
                "field_descriptions_tuple": tuple(field_descriptions_list), # Use tuple for hashability
                "field_data": np.array(all_field_data, dtype=np.uint32), # N x num_fields
                "filepath": filepath
            }

    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    except struct.error as e:
        print(f"Error: Could not unpack binary data from {filepath}. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return None

def find_and_parse_files(directory="."):
    """
    Finds files matching the pattern in the given directory and parses them.
    """
    parsed_files_data = []
    for item in os.listdir(directory):
        filepath = os.path.join(directory, item)
        if os.path.isfile(filepath):
            filename_info = parse_filename(item)
            if filename_info:
                file_name_base, prob = filename_info
                print(f"Processing file: {item} (Base: {file_name_base}, Prob: {prob})")
                data = read_stats_file(filepath)
                if data:
                    data["file_name_base"] = file_name_base
                    data["prob"] = prob
                    # If num_fields from header differs from parsed descriptions, prioritize descriptions for data shape.
                    # This assumes descriptions correctly define the columns we care about.
                    if data["field_data"].shape[1] != len(data["field_descriptions_tuple"]) and data["field_data"].size > 0 :
                         print(f"Warning: Mismatch for {filepath}. Header num_fields: {data['num_fields_header']}, "
                               f"Parsed descriptions: {len(data['field_descriptions_tuple'])}. "
                               f"Field data shape: {data['field_data'].shape}. Adjusting data columns to match descriptions if possible.")
                         if data["field_data"].shape[1] > len(data["field_descriptions_tuple"]):
                             data["field_data"] = data["field_data"][:, :len(data["field_descriptions_tuple"])]
                         # If fewer columns in data than descriptions, plotting will be limited by data.

                    parsed_files_data.append(data)
    return parsed_files_data

def group_data(parsed_files_data):
    """
    Groups parsed file data by field_descriptions_tuple.
    """
    grouped = defaultdict(list)
    for data_item in parsed_files_data:
        # Only group if there are field descriptions.
        if data_item["field_descriptions_tuple"]:
            grouped[data_item["field_descriptions_tuple"]].append(data_item)
        else:
            print(f"Skipping grouping for {data_item['filepath']} due to empty field descriptions.")
    return grouped

def plot_grouped_data(grouped_data, file_name_bases_overall):
    """
    Generates a plot for each group of field descriptions.
    """
    if not grouped_data:
        print("No data to plot after grouping.")
        return

    unique_file_names = sorted(list(set(file_name_bases_overall)))
    marker_map = {name: MARKERS[i % len(MARKERS)] for i, name in enumerate(unique_file_names)}

    plot_index = 0
    for desc_tuple, data_list in grouped_data.items():
        if not desc_tuple: # Skip if for some reason description tuple is empty
            continue

        plt.figure(figsize=(14, 8)) # Create a new figure for each group
        ax = plt.gca()

        num_actual_fields_to_plot = len(desc_tuple) # Based on parsed descriptions
        title_str = f"Field Group: {', '.join(desc_tuple)}"
        print(f"\nGenerating plot for: {title_str} ({num_actual_fields_to_plot} fields)")

        plot_data_points_exist = False

        for data_item in data_list:
            prob = data_item["prob"]
            file_name_base = data_item["file_name_base"]
            marker = marker_map.get(file_name_base, 'x') # Default marker

            if data_item["field_data"].size == 0:
                # print(f"  No field data in {data_item['filepath']} for this group.")
                continue

            # Data has shape (N, num_fields_from_header)
            # We need to plot up to num_actual_fields_to_plot columns.
            # If N (number of samples/rows in one file) > 1, we average the values for that PROB.
            if data_item["field_data"].shape[0] > 0:
                # Take mean across N samples if N > 1. field_data has shape (N, fields_in_data)
                # We will only use the first 'num_actual_fields_to_plot' columns of this mean data.
                mean_field_values_for_file = np.mean(data_item["field_data"], axis=0)

                cols_to_plot = min(len(mean_field_values_for_file), num_actual_fields_to_plot)

                for field_idx in range(cols_to_plot):
                    value = mean_field_values_for_file[field_idx]
                    color = f'C{field_idx}'
                    field_label_name = desc_tuple[field_idx]

                    # Add to a temporary structure to sort by PROB later if plotting lines
                    # For now, we plot individual points, then connect if multiple PROBs exist for same FILE_NAME+Field
                    # To do this properly, collect all points first, then plot lines.

                    # For simplicity in this iteration, we'll plot point by point.
                    # To draw lines, we'd need to collect all (prob, value) for each (file_name_base, field_idx)
                    # then sort by prob and plot.
                    # Let's refine this for line plotting:

                    # This will be handled by the refined plotting logic below.
                    pass # Placeholder, main plotting logic is next

        # Refined plotting logic: Collect data then plot lines
        # Structure: { (file_name_base, field_idx): [(prob, value), ...], ... }
        series_data = defaultdict(list)

        for data_item in data_list:
            prob = data_item["prob"]
            file_name_base = data_item["file_name_base"]

            if data_item["field_data"].size == 0:
                continue

            plot_data_points_exist = True # Mark that we have something to plot for this group
            mean_field_values_for_file = np.mean(data_item["field_data"], axis=0)
            cols_to_plot = min(len(mean_field_values_for_file), num_actual_fields_to_plot)

            for field_idx in range(cols_to_plot):
                series_data[(file_name_base, field_idx)].append((prob, mean_field_values_for_file[field_idx]))

        if not plot_data_points_exist:
            print(f"No data points to plot for group: {title_str}")
            plt.close() # Close the figure if no data
            continue

        # Now, plot each series
        legend_handles = []
        # Keep track of labels to avoid duplicates for (marker for FILE_NAME, color for field_idx)
        # More simply, label by 'FILE_NAME - Field Description'

        plotted_labels = set()

        for (file_name_base, field_idx), points in series_data.items():
            if not points:
                continue

            points.sort() # Sort by PROB for line plotting
            probs = [p[0] for p in points]
            values = [p[1] for p in points]

            marker = marker_map.get(file_name_base, 'x')
            color = f'C{field_idx}'
            field_label_name = desc_tuple[field_idx]
            series_label = f'{file_name_base} - {field_label_name}'

            # Only add label if it's new, to prevent legend duplication if linestyle is also used
            if series_label not in plotted_labels:
                ax.plot(probs, values, marker=marker, linestyle='-', color=color, label=series_label)
                plotted_labels.add(series_label)
            else: # Plot without adding to legend again
                 ax.plot(probs, values, marker=marker, linestyle='-', color=color)


        ax.set_xlabel("PROB")
        ax.set_ylabel("Field Values (mean over samples per file)")
        ax.set_title(title_str, fontsize=10)

        # Shrink current axis's width and height to make room for legend
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        ax.grid(True)

        # plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend; alternative to set_position
        plot_filename = f"plot_group_{plot_index+1}_{'_'.join(desc_tuple[:2]).replace(' ','_').replace('/','_')[:30]}.png"
        plt.savefig(plot_filename)
        print(f"Saved plot as {plot_filename}")
        plot_index += 1

    if plot_index > 0:
        plt.show() # Show all generated figures at the end
    else:
        print("No plots were generated.")


# --- Main Execution ---
if __name__ == "__main__":
    target_directory = "." # Configure to your target directory
    print(f"Scanning directory: {os.path.abspath(target_directory)} using pattern: {FILE_PATTERN}")

    all_parsed_data = find_and_parse_files(target_directory)

    if not all_parsed_data:
        print("No files found or parsed successfully.")
    else:
        print(f"\nSuccessfully parsed {len(all_parsed_data)} files.")

        # Collect all unique FILE_NAME bases for marker mapping
        file_name_bases_overall = list(set(d["file_name_base"] for d in all_parsed_data if "file_name_base" in d))

        grouped_data = group_data(all_parsed_data)
        print(f"\nData grouped into {len(grouped_data)} groups based on field descriptions.")

        for i, (desc_tuple, data_items) in enumerate(grouped_data.items()):
            file_names_in_group = set(d['file_name_base'] for d in data_items)
            print(f"  Group {i+1}: Descriptions = {desc_tuple}, Num Files = {len(data_items)}, Sources = {file_names_in_group}")

        plot_grouped_data(grouped_data, file_name_bases_overall)
