import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("lyly99/logodet3k")

print("Path to dataset files:", path)
