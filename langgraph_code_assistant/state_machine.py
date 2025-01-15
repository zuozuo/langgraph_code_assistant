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
        """Generate code solution using CodeGenerator"""
        from .code_generator import CodeGenerator
        
        # Get current state
        messages = state["messages"]
        iterations = state["iterations"]
        
        # Initialize generator (you may want to pass model/provider from config)
        generator = CodeGenerator()
        
        # Generate solution using last user message
        last_user_message = next(msg for msg in reversed(messages) if msg[0] == "user")
        solution = generator.generate(context="", question=last_user_message[1])
        
        # Update state
        state["generation"] = solution
        state["iterations"] = iterations + 1
        
        # Add assistant response to messages
        state["messages"].append(("assistant", f"Generated solution:\n{solution.code}"))
        
        return state

    def _check_code(self, state: GraphState) -> dict:
        """Check code imports and execution using CodeEvaluator"""
        from .evaluator import CodeEvaluator
        
        # Get current state
        solution = state["generation"]
        evaluator = CodeEvaluator()
        
        # Evaluate solution
        evaluation = evaluator.evaluate_solution(solution)
        
        # Update state based on evaluation
        if not evaluation["imports_valid"]:
            state["error"] = "yes"
            state["messages"].append(("system", "Import check failed"))
        elif not evaluation["code_executes"]:
            state["error"] = "yes"
            state["messages"].append(("system", "Code execution failed"))
        else:
            state["error"] = "no"
            
        return state

    def _decide_next_step(self, state: GraphState) -> str:
        """Determine next step based on state"""
        if state["error"] == "no" or state["iterations"] >= self.max_iterations:
            return "end"
        return "retry"

    def compile(self):
        """Compile the state machine"""
        return self.workflow.compile()
