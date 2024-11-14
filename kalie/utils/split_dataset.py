import os
import json 
import random
random.seed(2023)
"""

Generates train, test, and validation jsons in archive_split
"""

def main(args):
    # Load JSON data
    with open(args.filepath, 'r') as file:
        data = json.load(file)

    # Create a list of tuples from the dictionary (key, value)
    items = list(data.items())
    
    # Shuffle the list of data points
    random.shuffle(items)
    
    # Calculate the split index
    train_ratio = 0.8
    split_index = int(len(items) * train_ratio)

    # Split the data into training and test datasets
    train_items = items[:split_index]
    test_items = items[split_index:]
    val_items = items[split_index:]

    # Convert lists back to dictionaries
    train_data = dict(train_items)
    test_data = dict(test_items)
    val_data = dict(val_items)

    outdir = args.outdir
    os.makedirs(f"{outdir}", exist_ok=True)
    # Save the 80% data to a new JSON file
    with open(f"{outdir}train.json", 'w') as train_file:
        json.dump(train_data, train_file, indent=4)

    # Save the 20% data to another JSON file
    with open(f"{outdir}test.json", 'w') as test_file:
        json.dump(test_data, test_file, indent=4)
        
    with open(os.path.join(f"{outdir}valid.json"), "w") as val_file:
        json.dump(val_data, val_file, indent=4)

"""
Usage:
python split_dataset.py --filepath ./archive/data.json --outpath ./archive_split
"""
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help='path to json where initial data is located')
    parser.add_argument("--outdir", type=str, help='directory where split data should be stored')
    args = parser.parse_args()
    main(args) 
