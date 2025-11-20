#this code is used to create a zip file of a workspace directory for uploading to colab
import os
import zipfile

def zip_workspace(workspace_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(workspace_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, workspace_dir)
                zipf.write(file_path, arcname)
    print(f"Workspace directory '{workspace_dir}' has been zipped into '{output_zip}'")

if __name__ == "__main__":
    workspace_directory = "C:\\Users\\SMest\\Dropbox\\nnGAN\\electrophysiology_transformer"  # Replace with your workspace directory path
    output_zip_file = "workspace.zip"  # Desired output zip file name
    zip_workspace(workspace_directory, output_zip_file)