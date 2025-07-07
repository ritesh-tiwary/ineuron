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
Task,Description
Create project repo,Initialize Git repository and folder structure
Define dependencies,"Create requirements.txt (pandas, sqlalchemy, loguru, cx_Oracle, pyodbc, etc.)"
Setup virtual environment,Use venv or poetry for isolation
Define BaseTask,"Abstract class for all ETL steps with run(), validate(), etc."
Implement TaskFactory,Dynamically create task instances based on type
Build TaskManager,Handle task orchestration via context manager
Implement config-driven orchestration,Use config.json or YAML to define ETL jobs
CSV Extractor,Read chunked CSV input
Database Extractor,Query Oracle or Sybase source systems (optional)
API Extractor,Add ability to extract from REST endpoints (optional)
Column drop/rename,Based on config
Type conversion,"str → int, float, datetime, etc."
Filtering,Row-level filters based on column conditions
Add derived columns,"Support for load_date, default values, etc."
Deduplication logic,Drop duplicates by primary key or full row
Oracle Load Task,Chunked or batched inserts with to_sql
Sybase Load Task,Use sqlalchemy + pyodbc with FreeTDS
CSV Writer,Write final result to local file
S3 Upload,Upload CSV to S3 bucket using boto3
Extract config,"extract_config.json (source type, file path, query, API)"
Transform config,"transform_config.json (rename, drop, filters, types)"
Load config,"load_config.json (DB destination, batch size, upsert, truncate)"
Unified schema,Optional: merge all 3 configs into one task config
Pydantic models for config,Validate JSON/YAML structure with helpful errors
Retry mechanism,Use tenacity or custom retry logic for DB inserts
Graceful error logging,Ensure failed chunks log clearly without killing the full pipeline
Centralized logger,"Use loguru for file+console logging, log per task run"
Resource cleanup,"Close DB connections, release memory chunks"
Temporary file handler,Auto-delete intermediate files after success
Unit tests,"For each Task type (extraction, transformation, load)"
Integration tests,Run full pipeline on small dummy dataset
Data validation post-load,"Row count, checksum, and sampling validations"
CLI Interface,Add argparse or Typer-based CLI to trigger ETL runs
Scheduled execution,"Use cron, APScheduler, or Celery beat for scheduled tasks"
Dockerize,Add Dockerfile for containerized execution
CI/CD,"GitHub Actions or similar for tests, linting, and packaging"
```
```
JIRA Epic: Secure FastAPI Services with Kong JWT Middleware

Description:
Develop a reusable, organization-wide FastAPI middleware that enforces authentication using JWT headers forwarded by Kong Gateway. This epic ensures all services consistently block unauthenticated access and provide authenticated user context through standard dependencies.
```
```
Thank you for reaching out. We’re aware of the issue and are actively working on a resolution. Our team is prioritizing this, and I’ll provide you with an update as soon as it’s resolved.

If there’s a specific deadline or additional details you’d like to share, please let me know, and we’ll do our best to accommodate.

Thank you for your patience.
```
