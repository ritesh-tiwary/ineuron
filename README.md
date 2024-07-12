# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
import boto3
import os
from tqdm import tqdm

def upload_file_multipart(file_path, bucket_name, object_name):
    s3_client = boto3.client('s3')
    file_size = os.path.getsize(file_path)
    part_size = 50 * 1024 * 1024  # 50 MB
    file_name = os.path.basename(file_path)

    # Create multipart upload
    multipart_upload = s3_client.create_multipart_upload(Bucket=bucket_name, Key=object_name)

    parts = []
    try:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name) as pbar:
            part_number = 1
            with open(file_path, 'rb') as file:
                while True:
                    data = file.read(part_size)
                    if not data:
                        break

                    part = s3_client.upload_part(
                        Bucket=bucket_name,
                        Key=object_name,
                        PartNumber=part_number,
                        UploadId=multipart_upload['UploadId'],
                        Body=data
                    )
                    parts.append({
                        'PartNumber': part_number,
                        'ETag': part['ETag']
                    })
                    part_number += 1
                    pbar.update(len(data))

        # Complete multipart upload
        s3_client.complete_multipart_upload(
            Bucket=bucket_name,
            Key=object_name,
            UploadId=multipart_upload['UploadId'],
            MultipartUpload={'Parts': parts}
        )
        print(f"Upload completed: {file_name}")

    except Exception as e:
        # Abort multipart upload in case of an error
        s3_client.abort_multipart_upload(
            Bucket=bucket_name,
            Key=object_name,
            UploadId=multipart_upload['UploadId']
        )
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    file_path = 'path/to/your/large/file'
    bucket_name = 'your-bucket-name'
    object_name = 'path/in/s3/where/file/will/be/stored'

    upload_file_multipart(file_path, bucket_name, object_name)

```
