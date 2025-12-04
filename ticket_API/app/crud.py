from sqlalchemy.orm import Session
from . import models, schemas

def get_ticket(db: Session, ticket_id: int):
    return db.query(models.Ticket).filter(models.Ticket.ticket_id == ticket_id).first()

def get_tickets(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Ticket).offset(skip).limit(limit).all()

def create_ticket(db: Session, ticket: schemas.TicketCreate):
    db_ticket = models.Ticket(**ticket.dict())
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    return db_ticket

def update_ticket(db: Session, ticket_id: int, ticket_update: schemas.TicketUpdate):
    db_ticket = db.query(models.Ticket).filter(models.Ticket.ticket_id == ticket_id).first()
    if db_ticket:
        update_data = ticket_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_ticket, key, value)
        db.commit()
        db.refresh(db_ticket)
    return db_ticket
