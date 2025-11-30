import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
# Base path to the data directory (where this script resides)
DATA_DIR = os.path.dirname(__file__)

# Path to the main CSV (we need it for the 'Target' column to stratify)
LABELS_CSV = os.path.join(DATA_DIR, 'stage_2_train_labels.csv')

# --- Input Folders (Source) ---
# These folders were created by the 02 and 03 scripts
SOURCE_IMAGES_FOLDER = os.path.join(DATA_DIR, 'processed-data', 'images')
SOURCE_LABELS_FOLDER = os.path.join(DATA_DIR, 'processed-data', 'labels')

# --- Output Folders (Destination) ---
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'processed-data', 'train', 'images')
TRAIN_LABELS_DIR = os.path.join(DATA_DIR, 'processed-data', 'train', 'labels')

VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'processed-data', 'val', 'images')
VAL_LABELS_DIR = os.path.join(DATA_DIR, 'processed-data', 'val', 'labels')

TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'processed-data', 'test', 'images')
TEST_LABELS_DIR = os.path.join(DATA_DIR, 'processed-data', 'test', 'labels')


# --- Split Configuration ---
# We'll do 70% train, 15% validation, 15% test
TEST_SET_SIZE = 0.15        # 15% of data for the final test set
VALIDATION_SET_SIZE = 0.15  # 15% of data for the validation set
RANDOM_SEED = 42            # Makes our "random" split reproducible
# ---------------------

def move_files(patient_id_list, source_img, source_lbl, dest_img, dest_lbl):
    """
    Moves a list of patient files (PNGs and TXTs) from
    their source folder to their destination folder.
    """
    # Create destination directories
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_lbl, exist_ok=True)
    
    file_count = 0
    # Use os.path.basename to get the final folder name (e.g., 'train', 'val', 'test')
    set_name = os.path.basename(os.path.dirname(dest_img))
    
    for patient_id in tqdm(patient_id_list, desc=f"Moving {set_name} files"):
        img_name = f"{patient_id}.png"
        txt_name = f"{patient_id}.txt"
        
        src_img_path = os.path.join(source_img, img_name)
        dest_img_path = os.path.join(dest_img, img_name)
        
        src_lbl_path = os.path.join(source_lbl, txt_name)
        dest_lbl_path = os.path.join(dest_lbl, txt_name)

        # Use copy instead of move to be safer, then we can delete the originals at the end
        # Let's stick with move to clear the source folders as planned
        try:
            # Move the image
            if os.path.exists(src_img_path):
                shutil.move(src_img_path, dest_img_path)
                file_count += 1
            else:
                print(f"Warning: Image file not found: {src_img_path}")
                
            # Move the label
            if os.path.exists(src_lbl_path):
                shutil.move(src_lbl_path, dest_lbl_path)
            else:
                print(f"Warning: Label file not found: {src_lbl_path}")
        except Exception as e:
            print(f"Error moving {patient_id}: {e}")
            
    return file_count

def split_data():
    """
    Performs a 3-way stratified split (train/val/test) and moves files.
    """
    print("Starting 3-way (train/val/test) data split...")
    
    try:
        df = pd.read_csv(LABELS_CSV)
    except FileNotFoundError:
        print(f"Error: Labels CSV not found at {LABELS_CSV}")
        return

    # We only need one row per patient to stratify
    # 'Target' (0 or 1) is our stratification key
    patient_df = df.drop_duplicates(subset=['patientId'])[['patientId', 'Target']]
    all_patient_ids = patient_df['patientId']
    all_labels = patient_df['Target']

    print(f"Found {len(all_patient_ids)} total patients.")
    print(f"Positive (Pneumonia) cases: {all_labels.sum()}")
    print(f"Negative cases: {len(all_patient_ids) - all_labels.sum()}")

    # --- Stratified Split (Step 1: Train/Val vs. Test) ---
    print(f"Performing first split (Train/Val vs. Test)...")
    
    # First, split into a large "train+val" set and a smaller "test" set
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        all_patient_ids,
        all_labels,
        test_size=TEST_SET_SIZE,
        stratify=all_labels,
        random_state=RANDOM_SEED
    )

    # --- Stratified Split (Step 2: Train vs. Val) ---
    print(f"Performing second split (Train vs. Validation)...")
    
    # Now we split the "train+val" set into the final train and val sets.
    # We must adjust the test_size percentage to be relative to the new, smaller set.
    val_size_adjusted = VALIDATION_SET_SIZE / (1.0 - TEST_SET_SIZE)
    
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_labels,
        test_size=val_size_adjusted,
        stratify=train_val_labels, # Stratify again!
        random_state=RANDOM_SEED
    )
    
    # --- Report Final Set Sizes ---
    total_count = len(all_patient_ids)
    train_count = len(train_ids)
    val_count = len(val_ids)
    test_count = len(test_ids)
    
    print("\n--- Split Ratios ---")
    print(f"Training Set:   {train_count} patients ({train_count/total_count:.1%})")
    print(f"Validation Set: {val_count} patients ({val_count/total_count:.1%})")
    print(f"Test Set:       {test_count} patients ({test_count/total_count:.1%})")
    print(f"Total:          {train_count + val_count + test_count} patients")


    # --- Move Files ---
    print("\nMoving training files...")
    move_files(
        train_ids,
        SOURCE_IMAGES_FOLDER,
        SOURCE_LABELS_FOLDER,
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR
    )
    
    print("Moving validation files...")
    move_files(
        val_ids,
        SOURCE_IMAGES_FOLDER,
        SOURCE_LABELS_FOLDER,
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR
    )
    
    print("Moving test files...")
    move_files(
        test_ids,
        SOURCE_IMAGES_FOLDER,
        SOURCE_LABELS_FOLDER,
        TEST_IMAGES_DIR,
        TEST_LABELS_DIR
    )

    print("\n--- Split Complete! ---")
    print(f"Original folders '{SOURCE_IMAGES_FOLDER}' and '{SOURCE_LABELS_FOLDER}' should now be empty.")


if __name__ == "__main__":
    try:
        import sklearn
    except ImportError:
        print("Missing required package: scikit-learn")
        print("Please install it: pip install scikit-learn")
        exit(1)
        
    split_data()