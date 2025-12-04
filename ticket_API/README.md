# Ticket Handling API

This is a FastAPI application for handling support tickets using SQLite.

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Run the server using uvicorn:
   ```bash
   uvicorn app.main:app --reload
   ```

2. The API will be available at `http://127.0.0.1:8000`.

## Documentation

FastAPI automatically generates interactive API documentation.

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## Database

The application uses SQLite. The database file `sql_app.db` will be created in the root directory upon the first run.

## API Endpoints

- `POST /tickets/`: Create a new ticket.
- `GET /tickets/`: List all tickets.
- `GET /tickets/{ticket_id}`: Get a specific ticket.
- `PUT /tickets/{ticket_id}`: Update a ticket.
