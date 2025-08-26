# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
class ObjectWrapper:
    """Encapsulates S3 object actions."""

    def __init__(self, s3_object):
        """
        :param s3_object: A Boto3 Object resource. This is a high-level resource in Boto3
                          that wraps object actions in a class-like structure.
        """
        self.object = s3_object
        self.key = self.object.key

---
    def copy(self, dest_object):
        """
        Copies the object to another bucket.

        :param dest_object: The destination object initialized with a bucket and key.
                            This is a Boto3 Object resource.
        """
        try:
            dest_object.copy_from(
                CopySource={"Bucket": self.object.bucket_name, "Key": self.object.key}
            )
            dest_object.wait_until_exists()
            logger.info(
                "Copied object from %s:%s to %s:%s.",
                self.object.bucket_name,
                self.object.key,
                dest_object.bucket_name,
                dest_object.key,
            )
        except ClientError:
            logger.exception(
                "Couldn't copy object from %s/%s to %s/%s.",
                self.object.bucket_name,
                self.object.key,
                dest_object.bucket_name,
                dest_object.key,
            )
            raise
```
```
import json
import pandas as pd
import re
import os
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from datetime import datetime

class GenericJsonParser:
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        self.schema = defaultdict(set)
        self.processed_tables = {}
        self.error_log = []
    
    def load_json(self, file_path: str) -> bool:
        """Load and repair potentially broken JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply common JSON repairs
            repairs = [
                (r',(\s*[}\]])', r'\1'),  # Remove trailing commas
                (r'//.*?\n', ''),  # Remove JavaScript-style comments
                (r'(\s*"[^"]+"\s*:\s*)"([^"]+)"\s*([,}\]])', r'\1"\2"\3'),  # Fix unquoted values
                (r'"\s*\+\s*"', ''),  # Remove string concatenation
                (r'[\x00-\x1f\x7f-\x9f]', ' ')  # Remove control characters
            ]
            
            for pattern, replacement in repairs:
                content = re.sub(pattern, replacement, content)
            
            # Ensure valid JSON structure
            if not content.strip().endswith(('}', ']')):
                content = content.rsplit(',', 1)[0] + ('}' if '{' in content else ']')
            
            self.raw_data = json.loads(content)
            return True
        except json.JSONDecodeError as e:
            self._log_error('JSON_LOAD_ERROR', f"Failed to parse JSON: {str(e)}")
            return False
    
    def analyze_schema(self, data: Any, path: str = '') -> None:
        """Recursively analyze JSON schema to detect structure"""
        if isinstance(data, dict):
            for key, value in data.items():
                full_path = f"{path}.{key}" if path else key
                if value is not None:
                    self.schema[full_path].add(self._get_type(value))
                self.analyze_schema(value, full_path)
        elif isinstance(data, list) and data:
            self.analyze_schema(data[0], path)
    
    def clean_value(self, value: Any) -> Any:
        """Generic value cleaner"""
        if isinstance(value, str):
            # Clean numeric strings
            if re.match(r'^-?\d+[,.]\d+$', value):
                try:
                    return float(value.replace(',', ''))
                except ValueError:
                    pass
            
            # Clean date strings
            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y', '%d-%b-%y'):
                try:
                    return datetime.strptime(value, fmt).date().isoformat()
                except (ValueError, TypeError):
                    continue
            
            # Clean embedded JSON
            if value.startswith(('{', '[')) and value.endswith(('}', ']')):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            
            return value.strip()
        
        elif isinstance(value, (int, float)):
            return value
        elif isinstance(value, (dict, list)):
            return self.clean_structure(value)
        else:
            return value
    
    def clean_structure(self, data: Any) -> Any:
        """Recursively clean JSON structure"""
        if isinstance(data, dict):
            return {k: self.clean_value(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self.clean_value(item) for item in data if item is not None]
        else:
            return self.clean_value(data)
    
    def normalize_to_tables(self, data: Any, parent_key: str = '', parent_id: str = None) -> None:
        """
        Recursively normalize JSON data to relational tables
        """
        if isinstance(data, dict):
            # Create table for current level
            table_name = parent_key.split('.')[-1] if parent_key else 'root'
            table_data = {
                '_parent_id': parent_id,
                **{k: [v] for k, v in data.items() if not isinstance(v, (dict, list))}
            }
            
            # Process nested structures
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    nested_key = f"{parent_key}.{key}" if parent_key else key
                    self.normalize_to_tables(value, nested_key, parent_id)
            
            self._add_to_table(table_name, table_data)
            
        elif isinstance(data, list):
            # Handle arrays of objects
            for item in data:
                if isinstance(item, (dict, list)):
                    self.normalize_to_tables(item, parent_key, parent_id)
    
    def process(self, file_path: str) -> bool:
        """Main processing pipeline"""
        if not self.load_json(file_path):
            return False
        
        self.analyze_schema(self.raw_data)
        self.clean_data = self.clean_structure(self.raw_data)
        self.normalize_to_tables(self.clean_data)
        return True
    
    def save_tables(self, output_dir: str = 'output') -> Dict[str, pd.DataFrame]:
        """Save all processed tables to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        for table_name, data in self.processed_tables.items():
            df = pd.DataFrame(data)
            if not df.empty:
                df.to_csv(os.path.join(output_dir, f'{table_name}.csv'), index=False)
        
        # Save schema information
        schema_df = pd.DataFrame([
            {'path': path, 'types': ', '.join(types)}
            for path, types in self.schema.items()
        ])
        schema_df.to_csv(os.path.join(output_dir, '_schema.csv'), index=False)
        
        # Save error log if any
        if self.error_log:
            pd.DataFrame(self.error_log).to_csv(
                os.path.join(output_dir, '_errors.csv'), 
                index=False
            )
        
        return self.processed_tables
    
    def _add_to_table(self, table_name: str, data: Dict[str, List]) -> None:
        """Add data to a table, extending columns as needed"""
        if table_name not in self.processed_tables:
            self.processed_tables[table_name] = defaultdict(list)
        
        # Ensure all existing columns are present
        for col in self.processed_tables[table_name]:
            data.setdefault(col, [None])
        
        # Add new columns
        for col, values in data.items():
            self.processed_tables[table_name][col].extend(values)
    
    def _get_type(self, value: Any) -> str:
        """Get simplified type name for schema analysis"""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, dict):
            return 'object'
        elif isinstance(value, list):
            return 'array'
        else:
            return 'unknown'
    
    def _log_error(self, error_type: str, message: str, context: Any = None) -> None:
        """Log processing errors"""
        self.error_log.append({
            'error_type': error_type,
            'message': message,
            'context': str(context)[:200] if context else None,
            'timestamp': datetime.now().isoformat()
        })


# Example Usage
if __name__ == "__main__":
    # Initialize parser
    parser = GenericJsonParser()
    
    # Process any JSON file
    if parser.process('financial_data_broken.json'):
        # Save normalized tables
        tables = parser.save_tables()
        print(f"Generated {len(tables)} tables:")
        print("\n".join(tables.keys()))
    else:
        print("Failed to process JSON file")
```

```
# base_task.py
# ------------
from abc import ABC, abstractmethod

class BaseTask(ABC):
    def __init__(self, config: dict):
        self.config = config

    def run(self):
        self.validate()
        self.execute()
        self.prepare()
        self.cleanup()

    @abstractmethod
    def validate(self):...

    @abstractmethod
    def execute(self):...

    # optional override	
    def prepare(self):...

    def cleanup(self):...

# extract_task.py
# -----------------
class ExtractTask(BaseTask):
    def validate(self):
        # Validate DB config, data source
        pass

    def execute(self):
        # Load data into DB using SQLAlchemy / psycopg2 / etc.
        pass

# transform_task.py
# -----------------
class TransformTask(BaseTask):
    def validate(self):
        # Validate input file, schema, transformations
        pass

    def execute(self):
        # Convert file format, alter rows/cols using Pandas
        pass

# load_task.py
# -----------------
class LoadTask(BaseTask):
    def validate(self):
        # Validate DB config, data source
        pass

    def execute(self):
        # Load data into DB using SQLAlchemy / psycopg2 / etc.
        pass


import pandas as pd
from sqlalchemy import create_engine
from base_task import BaseTask


class CsvToOracleTask(BaseTask):
    def validate(self):
        required_keys = ['csv_path', 'oracle_conn_str', 'table_name']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")

    def execute(self):
        csv_path = self.config['csv_path']
        oracle_conn_str = self.config['oracle_conn_str']
        table_name = self.config['table_name']
        chunk_size = self.config.get('chunk_size', 5000)

        print(f"Loading CSV from {csv_path} to Oracle table {table_name}...")

        df_iter = pd.read_csv(csv_path, chunksize=chunk_size)
        engine = create_engine(oracle_conn_str)

        for i, chunk in enumerate(df_iter):
            print(f"Inserting chunk {i + 1}")
            chunk.to_sql(table_name, con=engine, if_exists='append', index=False, method='multi')

        print("CSV load to Oracle complete.")

# task_factory.py
# ---------------
from data_load_task import CsvToOracleTask
from base_task import BaseTask

class TaskFactory:
    @staticmethod
    def create(task_type: str, config: dict) -> BaseTask:
        if task_type == "csv_to_oracle":
            return CsvToOracleTask(config)
        raise ValueError(f"Unsupported task type: {task_type}")

# task_manager.py
# ---------------
from task_factory import TaskFactory
from loguru import logger

class TaskManager:
    def __init__(self, task_configs: list[dict]):
        self.task_configs = task_configs
        self.active_tasks = []

    def __enter__(self):
        logger.info("TaskManager context initialized.")
        # Placeholder: You can initialize shared resources here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("TaskManager context exiting. Cleaning up...")
        # Placeholder: Clean up shared resources if needed
        if exc_type:
            logger.error(f"Exception: {exc_type.__name__}, Message: {exc_val}")
        return False  # Don't suppress exceptions

    def run(self):
        for conf in self.task_configs:
            task = TaskFactory.create(conf["type"], conf)
            self.active_tasks.append(task)
            try:
                logger.info(f"Running task of type: {conf['type']}")
                task.execute()
            except Exception as e:
                logger.exception(f"Task of type {conf['type']} failed: {e}")


# main.py
# -------
import json
from task_manager import TaskManager

def main():
    with open('config.json') as f:
        task_configs = json.load(f)

     with TaskManager(task_configs) as t:
        t.run()

if __name__ == "__main__":
    main()

# config.json
# -----------
[
  {
    "type": "csv_to_oracle",
    "csv_path": "input_data.csv",
    "oracle_conn_str": "oracle+cx_oracle://username:password@hostname:port/service_name",
    "table_name": "TARGET_TABLE",
    "chunk_size": 1000
  }
]

# extract_config.json
# -------------------
{
  "source_type": "csv",
  "csv": {
    "path": "input_data.csv",
    "encoding": "utf-8",
    "delimiter": ",",
    "header": true,
    "chunk_size": 1000
  },
  "database": {
    "enabled": false,
    "conn_str": "oracle+cx_oracle://username:password@host:1521/service",
    "query": "SELECT * FROM source_table"
  },
  "api": {
    "enabled": false,
    "url": "https://api.example.com/data",
    "headers": {
      "Authorization": "Bearer <your-token>"
    },
    "params": {
      "start_date": "2024-01-01",
      "end_date": "2024-01-31"
    }
  }
}

# transform_config.json
# -------------------
{
  "enabled": true,
  "drop_columns": ["unwanted_column"],
  "rename_columns": {
    "old_name": "new_name",
    "amount": "total_amount"
  },
  "add_columns": [
    {
      "name": "load_date",
      "default": "NOW"  // could be "NOW", "STATIC", or a value
    }
  ],
  "filter_conditions": [
    {
      "column": "amount",
      "operator": ">",
      "value": 100
    }
  ],
  "column_types": {
    "id": "int",
    "total_amount": "float"
  },
  "deduplicate_on": ["id"]
}

# load_config.json
# ----------------
{
  "destination_type": "oracle",
  "oracle": {
    "conn_str": "oracle+cx_oracle://username:password@localhost:1521/orclpdb1",
    "table_name": "TARGET_TABLE",
    "write_mode": "append",  // or "replace", "upsert"
    "batch_size": 1000,
    "parallel": true,
    "upsert_keys": ["id"],
    "truncate_before_load": false
  },
  "csv": {
    "enabled": false,
    "path": "output_data.csv",
    "index": false
  },
  "s3": {
    "enabled": false,
    "bucket": "my-bucket",
    "key": "output/etl.csv",
    "aws_access_key": "xxx",
    "aws_secret_key": "yyy"
  }
}
```
```

# app.py
import asyncio
import hashlib
import hmac
import queue
from time import time
from typing import Iterator, Optional

import boto3
from fastapi import FastAPI, Request, Header, HTTPException
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

# >>> configure
bucket_name = "your-bucket"
s3_client = boto3.client("s3")

# 🔒 whitelist secure algorithms; reject md5/sha1
ALLOWED_ALGOS = {"sha256", "blake2b"}


def upload_part(bucket: str, key: str, upload_id: str, part_number: int, data: bytes):
    """
    Blocking call that uploads a single part to S3 and returns the descriptor needed by CompleteMultipartUpload.
    """
    response = s3_client.upload_part(
        Bucket=bucket,
        Key=key,
        PartNumber=part_number,
        UploadId=upload_id,
        Body=data,  # immutable bytes per part to avoid buffer mutation races
    )
    return {"ETag": response["ETag"], "PartNumber": part_number}


def _iter_from_queue(q: "queue.Queue[Optional[bytes]]") -> Iterator[bytes]:
    """
    Blocking iterator that yields bytes from a Queue until it receives None (sentinel).
    """
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk


def s3_multipart_upload_with_hash_from_iter(
    byte_iter: Iterator[bytes],
    object_key: str,
    expected_hash: str,
    part_size: int = 10 * 1024 * 1024,
    max_workers: int = 5,
    algo: str = "sha256",
) -> str:
    """
    Blocking function:
      - Consumes an iterator of bytes (streamed from FastAPI request),
      - Computes secure digest over the entire payload,
      - Performs S3 multipart upload in parallel,
      - Verifies checksum and completes/aborts accordingly.
    """
    if algo not in ALLOWED_ALGOS:
        raise ValueError(f"Unsupported or insecure algorithm: {algo}")

    h = hashlib.new(algo)
    response = s3_client.create_multipart_upload(Bucket=bucket_name, Key=object_key)
    upload_id = response["UploadId"]

    parts = []
    part_number = 1
    futures = []

    # We’ll fill parts up to `part_size` using a reusable buffer
    buf = bytearray()
    executor = ThreadPoolExecutor(max_workers=max_workers)

    try:
        for chunk in byte_iter:
            if not chunk:
                continue

            # Zero-copy view of incoming chunk
            mv = memoryview(chunk)
            try:
                # Update digest once for the whole chunk
                h.update(mv)

                # Fill upload buffer to part boundaries
                offset = 0
                mv_len = len(mv)
                while offset < mv_len:
                    remaining = part_size - len(buf)
                    if remaining <= 0:
                        # Shouldn't happen; safety guard
                        data = bytes(buf)
                        futures.append(
                            executor.submit(
                                upload_part,
                                bucket_name,
                                object_key,
                                upload_id,
                                part_number,
                                data,
                            )
                        )
                        part_number += 1
                        buf.clear()
                        remaining = part_size

                    take = min(remaining, mv_len - offset)
                    # Extend buffer from current chunk slice
                    buf.extend(mv[offset : offset + take])
                    offset += take

                    # If buffer reached a full part, freeze and submit
                    if len(buf) == part_size:
                        data = bytes(buf)  # materialize immutable payload
                        futures.append(
                            executor.submit(
                                upload_part,
                                bucket_name,
                                object_key,
                                upload_id,
                                part_number,
                                data,
                            )
                        )
                        part_number += 1
                        buf.clear()
            finally:
                # ✅ Always release the memoryview
                mv.release()

        # Final (possibly partial) part
        if buf:
            data = bytes(buf)
            futures.append(
                executor.submit(
                    upload_part, bucket_name, object_key, upload_id, part_number, data
                )
            )
            part_number += 1
            buf.clear()

        # Collect uploaded parts
        for fut in as_completed(futures):
            parts.append(fut.result())

        # Stop the executor
        executor.shutdown(wait=True)

        # Sort by PartNumber as AWS requires ordered list
        parts.sort(key=lambda x: x["PartNumber"])

        # 🔒 Validate checksum (constant-time)
        calculated = h.hexdigest()
        if not hmac.compare_digest(calculated, expected_hash.lower()):
            s3_client.abort_multipart_upload(
                Bucket=bucket_name, Key=object_key, UploadId=upload_id
            )
            raise ValueError(
                f"Checksum mismatch! Expected: {expected_hash}, Calculated: {calculated}"
            )

        # Complete upload
        s3_client.complete_multipart_upload(
            Bucket=bucket_name,
            Key=object_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

        # Optional: tag the object with the checksum (algo key)
        try:
            s3_client.put_object_tagging(
                Bucket=bucket_name,
                Key=object_key,
                Tagging={"TagSet": [{"Key": algo, "Value": calculated}]},
            )
        except Exception:
            # tagging failure shouldn't fail the upload
            pass

        return calculated

    except Exception:
        # Abort on any error (best-effort)
        try:
            s3_client.abort_multipart_upload(
                Bucket=bucket_name, Key=object_key, UploadId=upload_id
            )
        except Exception:
            pass
        raise
    finally:
        # Ensure no stray threads
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


@app.post("/upload")
async def upload(
    request: Request,
    filename: str = Header(..., alias="X-Filename"),
    content_digest: str = Header(..., alias="X-Content-Digest"),  # hex digest expected
    algo: str = Header("sha256", alias="X-Hash-Alg"),  # e.g., "sha256" or "blake2b"
):
    """
    Streaming upload endpoint:
      - Producer: async reads request.stream() and pushes chunks to a queue (with backpressure).
      - Consumer: blocking S3 multipart uploader runs in a worker thread pulling from the queue.
      - Computes secure checksum (algo whitelisted) and verifies against client-provided digest.
    """
    start = time()

    # Backpressure-friendly buffer between async stream and blocking uploader
    q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=50)

    async def producer():
        try:
            async for chunk in request.stream():
                if not chunk:
                    continue
                # Non-blocking enqueue with cooperative backoff (don't block the event loop)
                while True:
                    try:
                        q.put_nowait(chunk)
                        break
                    except queue.Full:
                        await asyncio.sleep(0.005)
        finally:
            # Always enqueue sentinel
            while True:
                try:
                    q.put_nowait(None)
                    break
                except queue.Full:
                    await asyncio.sleep(0.005)

    # Kick off the producer
    prod_task = asyncio.create_task(producer())

    # Run the blocking S3 uploader in a default thread pool
    loop = asyncio.get_running_loop()
    try:
        digest: str = await loop.run_in_executor(
            None,
            lambda: s3_multipart_upload_with_hash_from_iter(
                _iter_from_queue(q),
                object_key=filename,
                expected_hash=content_digest,
                part_size=10 * 1024 * 1024,
                max_workers=5,
                algo=algo,
            ),
        )
    except ValueError as ve:
        await prod_task
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        await prod_task
        raise HTTPException(status_code=500, detail=f"Upload failed: {e!s}")

    # Ensure producer ended cleanly (queue got sentinel)
    await prod_task

    end = time()
    return {
        "method": "multipart-parallel-digest",
        "duration_sec": round(end - start, 2),
        "algo": algo,
        "digest": digest,
    }

```

```
import paramiko
import socket
import logging
from contextlib import contextmanager
from time import sleep


class SFTPClient:
    def __init__(self, host, port=22, username=None, password=None, pkey=None, 
                 key_filename=None, timeout=10, retries=3, retry_delay=5, logger=None):
        """
        A safe wrapper around Paramiko's SFTPClient.

        Args:
            host (str): SFTP server hostname or IP.
            port (int): SFTP server port (default 22).
            username (str): SSH username.
            password (str): SSH password (optional if using key).
            pkey (str): Paramiko PKey object for key auth.
            key_filename (str): Path to private key file.
            timeout (int): Connection timeout in seconds.
            retries (int): Retry attempts on failure.
            retry_delay (int): Delay between retries (seconds).
            logger (logging.Logger): Optional logger instance.
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.pkey = pkey
        self.key_filename = key_filename
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger(__name__)
        
        self.transport = None
        self.sftp = None

    def connect(self):
        """Establish SFTP connection with retries."""
        attempt = 0
        while attempt < self.retries:
            try:
                self.logger.info(f"Connecting to {self.host}:{self.port} (Attempt {attempt+1})")
                transport = paramiko.Transport((self.host, self.port))
                transport.connect(
                    username=self.username,
                    password=self.password,
                    pkey=self.pkey,
                    hostkey=None
                )
                self.transport = transport
                self.sftp = paramiko.SFTPClient.from_transport(transport)
                self.logger.info("SFTP connection established")
                return self
            except (paramiko.SSHException, socket.error) as e:
                attempt += 1
                self.logger.warning(f"SFTP connection failed: {e}. Retrying in {self.retry_delay}s...")
                sleep(self.retry_delay)
        raise ConnectionError(f"Failed to connect to {self.host} after {self.retries} attempts")

    def close(self):
        """Close SFTP and transport cleanly."""
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.transport:
            self.transport.close()
            self.transport = None
        self.logger.info("SFTP connection closed")

    @contextmanager
    def session(self):
        """Context manager for auto-connect and cleanup."""
        try:
            self.connect()
            yield self
        finally:
            self.close()

    def upload(self, local_path, remote_path):
        """Upload a file to the SFTP server."""
        self.logger.info(f"Uploading {local_path} → {remote_path}")
        self.sftp.put(local_path, remote_path)
        self.logger.info("Upload complete")

    def download(self, remote_path, local_path):
        """Download a file from the SFTP server."""
        self.logger.info(f"Downloading {remote_path} → {local_path}")
        self.sftp.get(remote_path, local_path)
        self.logger.info("Download complete")

    def listdir(self, path="."):
        """List directory contents on SFTP server."""
        return self.sftp.listdir(path)

    def remove(self, remote_path):
        """Remove a file from SFTP server."""
        self.logger.info(f"Removing {remote_path}")
        self.sftp.remove(remote_path)
        self.logger.info("File removed")

```

```
Thank you for reaching out. We’re aware of the issue and are actively working on a resolution. Our team is prioritizing this, and I’ll provide you with an update as soon as it’s resolved.

If there’s a specific deadline or additional details you’d like to share, please let me know, and we’ll do our best to accommodate.

Thank you for your patience.
```
