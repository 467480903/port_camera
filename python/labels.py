import os
import glob

def modify_labels():
    # Path to the labels directory
    labels_dir = "/home/yy/port_camera/yoloTrain4/val/labels/"
    
    # Find all files matching the pattern all_*.txt
    pattern = os.path.join(labels_dir, "*_*.txt")
    txt_files = glob.glob(pattern)
    
    print(f"Found {len(txt_files)} files to process")
    
    # Process each file
    for file_path in txt_files:
        modified = False
        new_lines = []
        
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                new_lines.append("")
                continue
                
            # Split by space
            parts = line.split()
            
            # Check if we have at least one element
            if len(parts) > 0:
                # Check if first item is '1'
                if parts[0] == '1':
                    # Modify it to '0'
                    parts[0] = '0'
                    modified = True
                
                # Join back with spaces
                new_line = " ".join(parts)
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        # Save the modified content back to the file if changes were made
        if modified:
            with open(file_path, 'w') as file:
                file.write("\n".join(new_lines))
            print(f"Modified: {os.path.basename(file_path)}")
        else:
            print(f"No changes needed: {os.path.basename(file_path)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    modify_labels()