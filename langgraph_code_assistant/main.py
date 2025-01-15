import typer
from pathlib import Path
from .code_generator import CodeGenerator
from .state_machine import CodeAssistantStateMachine
from .evaluator import CodeEvaluator

app = typer.Typer()

@app.command()
def generate(
    question: str = typer.Argument(help="Question to generate code for"),
    context_file: Path = typer.Option(..., "--context-file", "-c", help="Path to context documentation"),
    model: str = typer.Option("openai/gpt-4o", "--model", "-m", help="Model to use for generation"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider (openai/anthropic)")
):
    """Generate code solution for a given question"""
    try:
        # Load context
        if not context_file.exists():
            typer.echo(f"Error: Context file '{context_file}' not found", err=True)
            raise typer.Exit(1)
            
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
        
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
