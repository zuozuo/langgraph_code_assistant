from typing import List, TypedDict
from langgraph.graph import END, StateGraph, START
from .code_generator import CodeSolution

class GraphState(TypedDict):
    """Represents the state of our graph"""
    error: str
    messages: List
    generation: CodeSolution
    iterations: int

class CodeAssistantStateMachine:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.workflow = StateGraph(GraphState)
        self._setup_graph()

    def _setup_graph(self):
        """Configure the state machine nodes and edges"""
        self.workflow.add_node("generate", self._generate_code)
        self.workflow.add_node("check_code", self._check_code)
        
        self.workflow.add_edge(START, "generate")
        self.workflow.add_edge("generate", "check_code")
        
        self.workflow.add_conditional_edges(
            "check_code",
            self._decide_next_step,
            {
                "end": END,
                "retry": "generate"
            }
        )

    def _generate_code(self, state: GraphState) -> dict:
        """Generate code solution"""
        # Implementation would use CodeGenerator
        return state

    def _check_code(self, state: GraphState) -> dict:
        """Check code imports and execution"""
        # Implementation would execute and validate code
        return state

    def _decide_next_step(self, state: GraphState) -> str:
        """Determine next step based on state"""
        if state["error"] == "no" or state["iterations"] >= self.max_iterations:
            return "end"
        return "retry"

    def compile(self):
        """Compile the state machine"""
        return self.workflow.compile()
