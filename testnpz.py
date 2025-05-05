import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import random
from scipy.ndimage import zoom

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Transformations
def interval_slice(arr, interval=2):
    """Extract slices from 3D volume at specified intervals."""
    return arr[::interval]


def normalize_mri(arr):
    """Normalize MRI to [0,1]."""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)




def process(file_path, case_id, matching_label):
    """Process a CT image and visualize the steps, skipping empty labels."""
    import nibabel as nib
    import numpy as np

    # Load NIfTI volumes
    img = nib.load(file_path)
    data = img.get_fdata()

    label_img = nib.load(matching_label)
    label = label_img.get_fdata()

    # Move axis (z to first) for consistency
    data = np.moveaxis(data, 2, 0)
    label = np.moveaxis(label, 2, 0)

    # Slice at intervals
    data = interval_slice(data, interval=4)
    label = interval_slice(label, interval=4)

    # Remove slices where label is all background (all zeros)
    valid_indices = [i for i in range(len(label)) if np.any(label[i] != 0)]
    if not valid_indices:
        print(f"[{case_id}] Skipped: All slices have background-only labels.")
        return None, None  # Skip this pair

    # Filter valid slices
    data = data[valid_indices]
    label = label[valid_indices]

    # Normalize CT data
    data_normalized = normalize_mri(data).astype(np.float32)

    return data_normalized, label

def extract_case_id(file_name):
    """Extract case ID from file name."""
    print(file_name)
    return file_name.split('.')[0]  # Case_00002_0000.nii.gz -> Case_00002_0000
def extract_ct_case_id(file_name):

    """Extract case ID from file name."""
    print(file_name)
    return file_name.split('_0000')[0]  # Case_00002_0000.nii.gz -> Case_00002_0000
def choose_output_base(
        output_root: str,
        case_id: str,
        modality: str,
        valid_ratio: float = 0.2
) -> str:


    out_dir = os.path.join(output_root, modality)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{case_id}_{modality}")


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='Process MRI or CT files and save as NPZ.')
    parser.add_argument('--mode', type=str, choices=['mr', 'ct'], required=True,
                        help='Processing mode: mr (MRI) or ct (CT)')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Path to the directory containing input images')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to the directory where processed files will be saved')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Path to the directory containing label files')
    args = parser.parse_args()
    input_path = Path(args.img_dir)
    output_path = Path(args.out_dir)
    label_path = Path(args.label_dir)
    print("Processing MRI files...")
    mri_files = list(Path(input_path).glob('*.nii.gz'))
    print(f"Found {len(mri_files)} MRI files in {input_path}")

    for mri_file in tqdm(mri_files):
        try:
            case_id = extract_ct_case_id(mri_file.name)
            matching_label = os.path.join(label_path, case_id + '.nii.gz')
            if os.path.exists(matching_label):
                print("Processing {0} with label {1}".format(mri_file, matching_label))
            else:
                print("find:",matching_label, mri_file)
                raise FileNotFoundError(f"Label file {matching_label} not found.")
            s,l = process(mri_file, case_id, matching_label)
            # output_base = choose_output_base(
            #     output_root=output_ct_path,
            #     case_id=case_id,
            #     modality='ct',
            #     valid_ratio=0.15
            # )
            output_base = output_path
            for i, slice in enumerate(s):
                height, width = slice.shape
                label = l[i]
                if height != 256 or width != 256:
                        zoom_factors = (256.0 / height, 256.0 / width)
                        slice = zoom(slice, zoom_factors, order=1)
                        label = zoom(label, zoom_factors, order=1)
                ''''NPZ file, where '[arr_0]' contains the image data, and '[arr_1]' contains the corresponding labels: '''
                data_dict = {"arr_0": slice, "arr_1": label} 
                print(slice.shape)
                print(label.shape)
                np.savez_compressed(f"{output_base}/{case_id}_s{i:04d}.npz", **data_dict)
        except Exception as e:
            print(f"Error processing {mri_file}: {e}")
    print(f"Done! Processed {len(mri_files)}  MRI files.")
    print(f"Results saved to {output_path}")