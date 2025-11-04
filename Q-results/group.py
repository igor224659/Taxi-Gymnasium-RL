import pathlib

# 1. Define the main folder (where the script is) and the output file name
main_folder = pathlib.Path.cwd()
output_filename = "combined_output.txt"
output_filepath = main_folder / output_filename

print(f"Script running in: {main_folder}")
print(f"Output will be saved to: {output_filepath}")

# 2. Open the output file in 'write' mode (this creates it or overwrites it)
with open(output_filepath, "w", encoding="utf-8") as outfile:
    
    # 3. Use rglob() to find ALL .txt files recursively (in all subfolders)
    for txt_file in main_folder.rglob("*.txt"):
        
        # 4. IMPORTANT: Skip the output file itself!
        if txt_file == output_filepath:
            continue
            
        # 5. IMPORTANT: Skip any .txt files in the main folder (only process subfolders)
        if txt_file.parent == main_folder:
            continue

        # 6. If we're here, it's a .txt file in a subfolder. Process it.
        print(f"Processing: {txt_file.relative_to(main_folder)}")
        try:
            # Read the content of the file
            content = txt_file.read_text(encoding="utf-8")
            
            # Write a separator (to know which file it came from)
            outfile.write(f"\n\n--- Content from: {txt_file.relative_to(main_folder)} ---\n\n")
            # Write the file's content
            outfile.write(content)
            
        except Exception as e:
            # Handle cases where the file might be unreadable (e.g., permission issues)
            error_message = f"  [ERROR] Could not read {txt_file.relative_to(main_folder)}. Error: {e}"
            print(error_message)
            outfile.write(f"\n\n--- FAILED to read: {txt_file.relative_to(main_folder)} (Error: {e}) ---\n\n")

print(f"\nDone! All text combined into {output_filename}")