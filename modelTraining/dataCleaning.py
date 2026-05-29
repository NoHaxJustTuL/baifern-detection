from fastai.vision.all import *

# Point this to your main dataset folder
path = Path('./modelTraining/dataset')

# Find all files and verify they are actually images
failed = verify_images(get_image_files(path))

# Delete any files that are corrupt or not real images
failed.map(Path.unlink)

print(f"Deleted {len(failed)} corrupted images.")