from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship
from .database import Base

class Customer(Base):
    __tablename__ = "customers"
    customer_id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    phone_number = Column(String)
    email = Column(String)
    address = Column(String)
    city = Column(String)
    pincode = Column(String)
    account_status = Column(String)
    join_date = Column(String)
    kyc_status = Column(String)

class IssueType(Base):
    __tablename__ = "issue_types"
    issue_type_id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)

class Agent(Base):
    __tablename__ = "agents"
    agent_id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    department = Column(String)
    shift_timing = Column(String)
    rating = Column(Float)

class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"))
    issue_type_id = Column(Integer, ForeignKey("issue_types.issue_type_id"))
    description = Column(String)
    status = Column(String)
    priority = Column(String)
    created_at = Column(String) # Storing as string as per requirement (TEXT)
    closed_at = Column(String, nullable=True)
    assigned_agent_id = Column(Integer, ForeignKey("agents.agent_id"), nullable=True)

    customer = relationship("Customer")
    issue_type = relationship("IssueType")
    agent = relationship("Agent")
