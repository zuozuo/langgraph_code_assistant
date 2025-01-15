import typer
from pathlib import Path
from .code_generator import CodeGenerator
from .state_machine import CodeAssistantStateMachine
from .evaluator import CodeEvaluator

app = typer.Typer()

@app.command()
def generate(
    question: str,
    context_file: Path = typer.Option(..., help="Path to context documentation"),
    model: str = typer.Option("gpt-4", help="Model to use for generation"),
    provider: str = typer.Option("openai", help="LLM provider (openai/anthropic)")
):
    """Generate code solution for a given question"""
    # Load context
    context = context_file.read_text()
    
    # Initialize components
    generator = CodeGenerator(model=model, provider=provider)
    evaluator = CodeEvaluator()
    state_machine = CodeAssistantStateMachine()
    
    # Generate solution
    solution = generator.generate(context, question)
    
    # Evaluate solution
    evaluation = evaluator.evaluate_solution(solution)
    
    # Output results
    typer.echo("\nGenerated Solution:")
    typer.echo(f"Description: {solution.prefix}")
    typer.echo(f"\nImports:\n{solution.imports}")
    typer.echo(f"\nCode:\n{solution.code}")
    
    typer.echo("\nEvaluation Results:")
    typer.echo(f"Imports valid: {evaluation['imports_valid']}")
    typer.echo(f"Code executes: {evaluation['code_executes']}")

if __name__ == "__main__":
    app()
