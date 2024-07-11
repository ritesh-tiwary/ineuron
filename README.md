# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
import os

def create_large_file_with_random_data(file_name, size_in_gb):
    """
    Create a large file of the specified size with random data.

    Parameters:
    file_name (str): The name of the file to be created.
    size_in_gb (int): The size of the file in gigabytes.
    """
    size_in_bytes = size_in_gb * 1024 * 1024 * 1024
    with open(file_name, 'wb') as f:
        chunk_size = 1024 * 1024  # 1MB chunks
        for _ in range(size_in_bytes // chunk_size):
            f.write(os.urandom(chunk_size))

if __name__ == "__main__":
    file_name = 'path/to/your/largefile.bin'  # The path to your large file
    size_in_gb = 20  # Size of the file in gigabytes

    create_large_file_with_random_data(file_name, size_in_gb)
    print(f"Created file {file_name} of size {size_in_gb} GB with random data")


```
