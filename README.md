# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
openapi: 3.0.0
info:
  title: Task Management API
  version: 1.0.0
paths:
  /tasks/{appid}/create:
    post:
      summary: Create a new task for a specific application
      parameters:
        - in: path
          name: appid
          required: true
          description: The ID of the application
          schema:
            type: integer
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                title:
                  type: string
                  description: The title of the task
                description:
                  type: string
                  description: Additional description of the task
                due_date:
                  type: string
                  format: date
                  description: The due date of the task
              required:
                - title
                - due_date
      responses:
        '201':
          description: Task created successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: integer
                    description: The ID of the created task
                  title:
                    type: string
                    description: The title of the created task
                  description:
                    type: string
                    description: Additional description of the created task
                  due_date:
                    type: string
                    format: date
                    description: The due date of the created task
        '400':
          description: Bad request. Missing or invalid parameters.
        '404':
          description: Application not found.
        '500':
          description: Internal server error. Something went wrong.

```
