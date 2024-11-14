import h5py
import json
import os
from PIL import Image

# OBJECTS = ["fur remover", "piece of fur", "cloth"]
# OBJECTS = ["cloth"]
# OBJECTS = ["usb stick", "computer"]
# OBJECTS = ["brush", "trash"]
# OBJECTS = ["open drawer"]
OBJECTS = ["scooper with small items in it", "bowl"]

def process_hdf5_to_json(hdf5_path, outdir, task_desc):
    # Path to the HDF5 file
    hdf5_file = h5py.File(hdf5_path, 'r')

    # Create directories for images if they do not exist
    img_folder = f'{outdir}imgs'
    os.makedirs(img_folder, exist_ok=True)

    # Dictionary to hold JSON data
    json_data_str = {}
    json_data_js = {} 

    # Iterate through each group in the HDF5 file
    for group_name in hdf5_file:
        group = hdf5_file[group_name]
        
        # Save the initial image
        img_data = group['init_image'][:]
        img = Image.fromarray(img_data)
        img_path = f"{img_folder}/{group_name}.png"
        img.save(img_path)
        
        # Prepare points data (string format)
        points_format_str = f"Grasp Point: {tuple(group['grasp_point'][:])}, " \
                        f"Function Point: {tuple(group['function_point'][:])}, " \
                        f"Target Point: {tuple(group['target_point'][:])}, " \
                        f"Pre-contact Point: {tuple(group['pre_contact_point'][:])}, " \
                        f"Post-contact Point: {tuple(group['post_contact_point'][:])}"
        
        # Add to JSON data
        json_data_str[group_name] = {
            "img": img_path,
            "points": points_format_str,
            "other": {
                "task": task_desc,
                "objects": OBJECTS
            }
        }

        # Prepare points data (json format)
        points_dict = {
            "Grasp Point": tuple(group['grasp_point'][:].tolist()),
            "Function Point": tuple(group['function_point'][:].tolist()),
            "Target Point": tuple(group['target_point'][:].tolist()),
            "Pre-contact Point": tuple(group['pre_contact_point'][:].tolist()),
            "Post-contact Point": tuple(group['post_contact_point'][:].tolist())
        }
        
        points_format_js = json.dumps(points_dict, indent=4)
        points_format_js = f"""{points_format_js}"""
        
        # Add to JSON data
        json_data_js[group_name] = {
            "img": img_path,
            "points": points_format_js,
            "other": {
                "task": task_desc,
                "objects": OBJECTS
            }
        }
    
    # Close the HDF5 file
    hdf5_file.close()

    # Write JSON data to file
    with open(f'{outdir}data_strformat.json', 'w') as json_file:
        json.dump(json_data_str, json_file, indent=4)

     # Write JSON data to file
    with open(f'{outdir}data_jsonformat.json', 'w') as json_file:
        json.dump(json_data_js, json_file, indent=4)

"""
Usage from root: 
python scripts/datacol/hdf5_to_json.py --task "Sweep the trash off the table with the brush." --filepath "./vlm_data/trash_sweeping/2024-05-02_50.hdf5" --outdir "./archive_2024-05-02/"
"""
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task description")
    parser.add_argument("--filepath", type=str, help="filepath to hdf5 file")
    parser.add_argument("--outdir", type=str, help="filepath to output directory")
    args = parser.parse_args()
    
    process_hdf5_to_json(args.filepath, args.outdir, args.task)
