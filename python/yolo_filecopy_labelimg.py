import os
import shutil

def copy_files(source_dir, labels_dir, images_dir, files):
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    for filename in files:
        txt_file = os.path.join(source_dir, filename)
        jpg_file = os.path.join(source_dir, filename.replace('.txt', '.jpg'))
        
        shutil.copy(txt_file, labels_dir)
        shutil.copy(jpg_file, images_dir)

def main():
    source_dir = '../label/left/'
    train_labels_dir = '../yoloTrain3/train/labels/'
    train_images_dir = '../yoloTrain3/train/images/'
    val_labels_dir = '../yoloTrain3/val/labels/'
    val_images_dir = '../yoloTrain3/val/images/'

    # Filter out classes.txt and only get actual annotation files
    txt_files = sorted([f for f in os.listdir(source_dir) 
                       if f.endswith('.txt') and f != 'classes.txt'])
    total_files = len(txt_files)
    
    train_split = int(0.8 * total_files)
    train_files = txt_files[:train_split]
    val_files = txt_files[train_split:]
    
    copy_files(source_dir, train_labels_dir, train_images_dir, train_files)
    copy_files(source_dir, val_labels_dir, val_images_dir, val_files)
    print(f"Copied {len(train_files)} files to train directory.")
    print(f"Copied {len(val_files)} files to val directory.")

if __name__ == "__main__":
    main()