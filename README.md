# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List
import json
import os
from datetime import datetime

app = FastAPI()

# Define the request and task model
class Task(BaseModel):
    request_id: str
    task_id: str

# Define the function to get the current month-year string
def get_current_month_year():
    now = datetime.now()
    return now.strftime("%Y-%m")

# Define the function to get the file path for the current month
def get_file_path():
    current_month_year = get_current_month_year()
    return os.path.join("data", f"{current_month_year}.json")

# Initialize the data directory
os.makedirs("data", exist_ok=True)

# Define the function to reset the file if it belongs to a previous year
def reset_file_if_necessary(file_path: str):
    if os.path.exists(file_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        current_year = datetime.now().year
        if file_mod_time.year != current_year:
            os.remove(file_path)  # Delete the old file

# Define the function to save tasks to a JSON file
def save_tasks_to_json(tasks: List[Task]):
    file_path = get_file_path()
    reset_file_if_necessary(file_path)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.extend([task.dict() for task in tasks])

    with open(file_path, "w") as f:
        json.dump(data, f)

# Define the endpoint to fetch tasks by request ID
@app.get("/gettasksbyrequestid/{request_id}")
async def get_tasks_by_request_id(request_id: str):
    file_path = get_file_path()
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="No data found for the current month")

    with open(file_path, "r") as f:
        data = json.load(f)

    tasks = [task for task in data if task["request_id"] == request_id]
    return tasks

# Define the endpoint to add tasks and store them in the background
@app.post("/addtasks")
async def add_tasks(tasks: List[Task], background_tasks: BackgroundTasks):
    background_tasks.add_task(save_tasks_to_json, tasks)
    return {"message": "Tasks are being processed"}

# Define the endpoint to fetch all data
@app.get("/fetchall")
async def fetch_all():
    file_path = get_file_path()
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="No data found for the current month")

    with open(file_path, "r") as f:
        data = json.load(f)

    return data

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

```
