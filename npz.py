import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from scipy.ndimage import zoom

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


valid_count = 0
train_count = 0

# input_path = "/mnt/d/Jiahe/Cornell/data"
# output_path = "/mnt/d/Jiahe/Cornell/data_npy"
# vis_path = "/mnt/d/Jiahe/Cornell/data_vis"

root_proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
root = os.path.dirname(os.path.realpath(__file__))

# input_path = f"{root}/data"
# output_path = f"{root}/data_npy"
# output_valid = f"{root}/data_npy/valid"
# output_train = f"{root}/data_npy/train"
print(root_proj)

ct_path = "/home/jwu235/bio/debug/data/ct/image"
ct_label_path = "/home/jwu235/bio/debug/data/ct/label"
mr_path = "/home/jwu235/bio/debug/data/mri"
output_ct_path = "./data/source"
output_mr_path = "./data/target"
# vis_path = f"{root_proj}/UDADiff_project/data_vis"/
output_valid = f"{root_proj}/UDADiff_project/UDADiffusion/data_npy/valid"
output_train = f"{root_proj}/UDADiff_project/UDADiffusion/data_npy/train"

# os.makedirs(output_valid, exist_ok=True)
# os.makedirs(output_train, exist_ok=True)


# Transformations
def interval_slice(arr, interval=2):
    """Extract slices from 3D volume at specified intervals."""
    return arr[::interval]


def normalize_ct(arr, hu_min=-250, hu_max=250):
    """Apply HU window to CT and normalize to [0,1]."""
    # HU clipping
    arr_clipped = np.clip(arr, hu_min, hu_max)
    arr_clipped = (arr_clipped - hu_min) / (hu_max - hu_min + 1e-8)
    return arr_clipped


def normalize_mri(arr):
    """Normalize MRI to [0,1]."""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


def visualize_processing(original, clipped, normalized, output_path, title='Processing Steps'):
    """Create visualization of original, clipped and normalized images."""
    # Choose a middle slice along the last dimension (axial)
    slice_idx = original.shape[2] // 2
    clipped_slice = None
    if clipped is not None:
        clipped_slice = clipped[:, :, slice_idx]

    # Extract slices from the third dimension
    original_slice = original[:, :, slice_idx]
    normalized_slice = normalized[:, :, slice_idx]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))

    # Plot original image
    im1 = axes[0].imshow(original_slice.T, cmap='gray', origin='lower', aspect='auto')
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot clipped image
    if clipped_slice is not None:
        im2 = axes[1].imshow(clipped_slice.T, cmap='gray', origin='lower', aspect='auto')
        axes[1].set_title('Clipped', fontsize=14)
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot normalized image
    im3 = axes[2].imshow(normalized_slice.T, cmap='gray', origin='lower', aspect='auto')
    axes[2].set_title('Normalized', fontsize=14)
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Add overall title
    plt.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.05)

    # Save figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# Processing functions
def process_ct(file_path, case_id, matching_label):
    """Process a CT image and visualize the steps."""
    # Load NIfTI file
    nifti_img = nib.load(file_path)
    ct_data = nifti_img.get_fdata()
    nifti_img = nib.load(file_path)
    ct_label = nifti_img.get_fdata()
    # Save original stats
    original_min = np.min(ct_data)
    original_max = np.max(ct_data)
    ct_label = np.moveaxis(ct_label,2,0)
    ct_data = np.moveaxis(ct_data, 2, 0)  # Move the last axis to the first position
    # Apply interval slice
    ct_data = interval_slice(ct_data, interval=4)
    ct_label = interval_slice(ct_label, interval=4)
    # Apply CT normalization
    ct_normalized = normalize_ct(ct_data).astype(np.float32)

    return ct_normalized, ct_label


def process_mri(file_path, case_id):
    """Process an MRI image and visualize the steps."""
    # Load NIfTI file
    nifti_img = nib.load(file_path)
    mri_data = nifti_img.get_fdata()

    # Save original stats
    original_min = np.min(mri_data)
    original_max = np.max(mri_data)

    # Get a copy of original data for visualization
    original_data = mri_data.copy()

    mri_data = np.moveaxis(mri_data, 2, 0)  # Move the last axis to the first position
    # Apply interval slice
    mri_data = interval_slice(mri_data, interval=4)

    # Apply MRI normalization
    mri_normalized = normalize_mri(mri_data).astype(np.float32)
    
    return mri_normalized


def extract_case_id(file_name):
    """Extract case ID from file name."""
    return file_name.split('.')[0]  # Case_00002_0000.nii.gz -> Case_00002_0000
def extract_ct_case_id(file_name):

    """Extract case ID from file name."""
    return file_name.split('_0000')[0]  # Case_00002_0000.nii.gz -> Case_00002_0000
def choose_output_base(
        output_root: str,
        case_id: str,
        modality: str,
        valid_ratio: float = 0.2
) -> str:
    global valid_count, train_count

    if modality not in ('mri', 'ct'):
        raise ValueError(f"modality must be 'mri' or 'ct', got {modality!r}")

    subset = 'valid' if random.random() < valid_ratio else 'train'
    # 更新计数器
    if subset == 'valid':
        valid_count += 1
    else:
        train_count += 1

    out_dir = os.path.join(output_root, subset, modality)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{case_id}_{modality}")


if __name__ == "__main__":
    print("Processing CT files...")
    ct_files = list(Path(ct_path).glob('*.nii.gz'))
    print(f"Found {len(ct_files)} CT files in {ct_path}")

    for ct_file in tqdm(ct_files):
        try:
            case_id = extract_ct_case_id(ct_file.name)
            matching_label = os.path.join(ct_label_path, case_id + '.nii.gz')
            if os.path.exists(matching_label):
                print("Processing {0} with label {1}".format(ct_file, matching_label))
            else:
                print("find:",matching_label, ct_file)
                raise FileNotFoundError(f"Label file {matching_label} not found.")
            s,l = process_ct(ct_file, case_id, matching_label)
            # output_base = choose_output_base(
            #     output_root=output_ct_path,
            #     case_id=case_id,
            #     modality='ct',
            #     valid_ratio=0.15
            # )
            output_base = output_ct_path
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
            print(f"Error processing {ct_file}: {e}")
    
    print("Processing MRI files...")
    mri_files = list(Path(mr_path).glob('*.nii.gz'))
    print(f"Found {len(mri_files)} MRI files in {mr_path}")

    for mri_file in tqdm(mri_files):
        try:
            case_id = extract_case_id(mri_file.name)
            
            processed = process_mri(mri_file, case_id)
            # output_base = choose_output_base(
            #     output_root=output_mr_path,
            #     case_id=case_id,
            #     modality='mri',
            #     valid_ratio=0.15
            # )
            output_base = output_mr_path
            for i, slice in enumerate(processed):
                height, width = slice.shape
                if height != 256 or width != 256:
                        zoom_factors = (256.0 / height, 256.0 / width)
                        slice = zoom(slice, zoom_factors, order=1)
                        label = zoom(label, zoom_factors, order=1)
                ''''NPZ file, where '[arr_0]' contains the image data, and '[arr_1]' contains the corresponding labels: '''
                label = np.zeros_like(slice)
                data_dict = {"arr_0" : slice, "arr_1" : label}
                print(slice.shape)
                print(label.shape)
                np.savez_compressed(f"{output_base}/{case_id}_s{i:04d}.npz", **data_dict)
        except Exception as e:
            print(f"Error processing {mri_file}: {e}")

    print(f"Done! Processed {len(ct_files)} CT files and {len(mri_files)} MRI files.")
    print(f"Results saved to {output_base} and {output_mr_path}")

    print(f"Total assigned to valid: {valid_count}")
    print(f"Total assigned to train: {train_count}")