from pydantic import BaseModel
from typing import Optional

class CustomerBase(BaseModel):
    name: str
    phone_number: str
    email: str
    address: str
    city: str
    pincode: str
    account_status: str
    join_date: str
    kyc_status: str

class CustomerCreate(CustomerBase):
    pass

class Customer(CustomerBase):
    customer_id: int
    class Config:
        orm_mode = True

class IssueTypeBase(BaseModel):
    name: str
    description: str

class IssueTypeCreate(IssueTypeBase):
    pass

class IssueType(IssueTypeBase):
    issue_type_id: int
    class Config:
        orm_mode = True

class AgentBase(BaseModel):
    name: str
    department: str
    shift_timing: str
    rating: float

class AgentCreate(AgentBase):
    pass

class Agent(AgentBase):
    agent_id: int
    class Config:
        orm_mode = True

class TicketBase(BaseModel):
    customer_id: int
    issue_type_id: int
    description: str
    priority: str
    status: str = "Open"
    created_at: str
    assigned_agent_id: Optional[int] = None
    closed_at: Optional[str] = None

class TicketCreate(TicketBase):
    pass

class TicketUpdate(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_agent_id: Optional[int] = None
    closed_at: Optional[str] = None

class Ticket(TicketBase):
    ticket_id: int

    class Config:
        orm_mode = True
