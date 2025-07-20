# main.py
import typer
import os
from core import run_isal_process # Assumes core returns (final_answer, intermediate_steps_dict)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax

app = typer.Typer(help="Project Chimera: Socratic Self-Debate with LLMs.")
console = Console()

@app.command()
def reason(
    prompt: str = typer.Argument(..., help="The initial prompt for the Socratic Arbitration Loop."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all intermediate reasoning steps."),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Your Gemini API key. Overrides GEMINI_API_KEY environment variable.", envvar="GEMINI_API_KEY")
):
    """
    Runs the Socratic Arbitration Loop on a given prompt.
    Requires GEMINI_API_KEY environment variable to be set, or passed via --api-key.
    """
    if not api_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY environment variable not set or --api-key not provided.")
        raise typer.Exit(code=1)
    
    console.print(Panel(Text("🤖 Starting Socratic Arbitration Loop...", justify="center"), style="bold blue"))
    
    final_answer, intermediate_steps = run_isal_process(prompt, api_key)
    
    if verbose:
        console.print(Panel(Text("--- Intermediate Steps ---", justify="center"), style="bold magenta"))
        for step_name, content in intermediate_steps.items():
            console.print(f"\n[bold cyan]### {step_name.replace('_', ' ').title()}:[/bold cyan]")
            # Use Syntax for code-like output, or just print for general text
            if "Output" in step_name or "Critique" in step_name or "Feedback" in step_name:
                console.print(Syntax(content, "markdown", theme="monokai", line_numbers=False, word_wrap=True))
            else:
                console.print(content)

    console.print(Panel(Text("--- Final Synthesized Answer ---", justify="center"), style="bold green"))
    console.print(Syntax(final_answer, "markdown", theme="monokai", line_numbers=False, word_wrap=True))

if __name__ == "__main__":
    app()