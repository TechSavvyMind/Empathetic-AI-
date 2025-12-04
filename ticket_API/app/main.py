from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Ticket Handling API", description="API for managing support tickets", version="1.0.0")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/tickets/", response_model=schemas.Ticket)
def create_ticket(ticket: schemas.TicketCreate, db: Session = Depends(get_db)):
    return crud.create_ticket(db=db, ticket=ticket)

@app.get("/tickets/", response_model=List[schemas.Ticket])
def read_tickets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    tickets = crud.get_tickets(db, skip=skip, limit=limit)
    return tickets

@app.get("/tickets/{ticket_id}", response_model=schemas.Ticket)
def read_ticket(ticket_id: int, db: Session = Depends(get_db)):
    db_ticket = crud.get_ticket(db, ticket_id=ticket_id)
    if db_ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return db_ticket

@app.put("/tickets/{ticket_id}", response_model=schemas.Ticket)
def update_ticket(ticket_id: int, ticket: schemas.TicketUpdate, db: Session = Depends(get_db)):
    db_ticket = crud.update_ticket(db, ticket_id=ticket_id, ticket_update=ticket)
    if db_ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return db_ticket
