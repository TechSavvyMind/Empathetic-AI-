import datetime
from sqlalchemy.orm import Session
from ticket_API.app import crud
from ticket_API.app import models
from ticket_API.app import schemas
from ticket_API.app.database import SessionLocal, engine

# Ensure Ticket DB tables exist (using your provided models.py)
models.Base.metadata.create_all(bind=engine)

def create_ticket_tool(description: str, priority: str = "Medium", issue_type_id: int = 1) -> str:
    """
    Tool function for AI to create a support ticket in the SQL database.
    
    Args:
        description (str): Detailed description of the issue.
        priority (str): 'Low', 'Medium', 'High', or 'Urgent'.
        issue_type_id (int): 1=General, 2=Technical, 3=Billing.
    
    Returns:
        str: Confirmation message with Ticket ID.
    """
    db: Session = SessionLocal()
    try:
        # Mocking customer ID (In a real app, pass this from the Orchestrator state)
        customer_id = 1 
        
        ticket_data = schemas.TicketCreate(
            customer_id=customer_id,
            issue_type_id=issue_type_id,
            description=description,
            priority=priority,
            status="Open",
            created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        ticket = crud.create_ticket(db=db, ticket=ticket_data)
        return f"✅ Ticket #{ticket.ticket_id} created successfully. Status: {ticket.status}, Priority: {ticket.priority}"
    except Exception as e:
        return f"❌ Failed to create ticket: {str(e)}"
    finally:
        db.close()