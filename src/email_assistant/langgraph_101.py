from typing import Literal, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from dotenv import load_dotenv
import uvicorn

load_dotenv()  # This will load from .env file in the project root

# Pydantic models for API requests/responses
class EmailRequest(BaseModel):
    message: str

class EmailResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
model_with_tools = llm.bind_tools([write_email], tool_choice="any")

def call_llm(state: MessagesState) -> MessagesState:
    """Run LLM"""
    output = model_with_tools.invoke(state["messages"])
    return {"messages": [output]}

def run_tool(state: MessagesState) -> MessagesState:
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        observation = write_email.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["run_tool", "__end__"]:
    """Route to tool handler, or end if Done tool called"""
    # Get the last message
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is a tool call, check if it's a Done tool call
    if last_message.tool_calls:
        return "run_tool"
    # Otherwise, we stop (reply to the user)
    return END

# Build the workflow
workflow = StateGraph(MessagesState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("run_tool", run_tool)
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue, {"run_tool": "run_tool", END: END})
workflow.add_edge("run_tool", END)

# Compile the workflow
langgraph_app = workflow.compile()

# Create FastAPI app
app = FastAPI(
    title="Email Assistant API",
    description="A LangGraph-powered email assistant API",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Email Assistant API is running"}

@app.post("/chat", response_model=EmailResponse)
async def chat_with_assistant(request: EmailRequest):
    """
    Send a message to the email assistant and get a response.
    The assistant can write emails using the write_email tool.
    """
    try:
        # Create initial state with user message
        initial_state = {"messages": [HumanMessage(content=request.message)]}
        
        # Run the workflow
        result = langgraph_app.invoke(initial_state)
        
        # Extract the final response
        messages = result["messages"]
        final_message = messages[-1]
        
        # Extract tool calls if any
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "tool": tool_call["name"],
                        "args": tool_call["args"]
                    })
        
        # Get the assistant's final response
        response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        return EmailResponse(
            response=response_content,
            tool_calls=tool_calls
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "email-assistant"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)