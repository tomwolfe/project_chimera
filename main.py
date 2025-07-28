# main.py
import typer
import os
from core import run_isal_process, TokenBudgetExceededError # Import new exception
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from llm_provider import GeminiProvider # Import to access cost calculation logic
from rich.syntax import Syntax

app = typer.Typer(help="Project Chimera: Socratic Self-Debate with LLMs.")
console = Console()

@app.command()
def reason(
    prompt: str = typer.Argument(..., help="The initial prompt for the Socratic Arbitration Loop."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all intermediate reasoning steps."),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Your Gemini API key. Overrides GEMINI_API_KEY environment variable.", envvar="GEMINI_API_KEY"),
    max_tokens_budget: int = typer.Option(10000, "--max-tokens", "-m", help="Maximum total tokens allowed for the entire process to control cost."),
    domain: str = typer.Option("auto", "--domain", "-d", help="Reasoning domain to use (auto, General, Science, Business, Creative)."),
    model_name: str = typer.Option("gemini-2.5-flash-lite", "--model", "-M", help="The LLM model to use (e.g., gemini-2.5-flash-lite, gemini-2.5-pro).")
):
    """
    Runs the Socratic Arbitration Loop on a given prompt.
    Requires GEMINI_API_KEY environment variable to be set, or passed via --api-key.
    """
    if not api_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY environment variable not set or --api-key not provided.")
        raise typer.Exit(code=1)
    
    # Initialize a dummy GeminiProvider to access cost calculation logic
    # This is a bit of a hack for CLI, as the real provider is initialized in core.py
    # A better design might be to move cost calculation to a utility function.
    # For now, this works.
    # temp_gemini_provider = GeminiProvider(api_key="dummy_key", model_name=model_name) # This line is not strictly needed for the current logic

    # CLI specific status update function
    def cli_status_callback(message: str, state: str = "running", expanded: bool = True,
                            current_total_tokens: int = 0, current_total_cost: float = 0.0,
                            estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
        # For CLI, we just print updates. The 'state' and 'expanded' are mostly for Streamlit.
        status_color = "blue"
        if state == "error":
            status_color = "red"
        elif state == "warning":
            status_color = "yellow"
        elif state == "complete":
            status_color = "green"

        console.print(f"[{status_color}]Status:[/{status_color}] {message}")
        console.print(f"  [bold]Tokens Used:[/bold] {current_total_tokens} | [bold]Estimated Cost:[/bold] ${current_total_cost:.4f}")
        if estimated_next_step_tokens > 0:
            budget_remaining = max_tokens_budget - current_total_tokens
            if estimated_next_step_tokens > budget_remaining:
                console.print(f"  [bold red]WARNING:[/bold red] Next step ({estimated_next_step_tokens} tokens / ${estimated_next_step_cost:.4f}) will exceed budget ({budget_remaining} remaining).")
            else:
                console.print(f"  [bold yellow]Next Step Estimate:[/bold] {estimated_next_step_tokens} tokens / ${estimated_next_step_cost:.4f} (Budget remaining: {budget_remaining})")
        console.print("-" * 80) # Separator for clarity

    # Initialize debate_instance to None before the try block
    debate_instance = None 
    try:
        console.print(Panel(Text(f"🤖 Starting Socratic Arbitration Loop (Budget: {max_tokens_budget} tokens)...", justify="center"), style="bold blue"))
        
        # First, get the debate instance
        debate_instance = run_isal_process(
            prompt, api_key, max_total_tokens_budget=max_tokens_budget,
            model_name=model_name, # Pass model name
            domain=domain, # Pass the selected domain
            streamlit_status_callback=cli_status_callback, # Pass CLI specific callback
            all_personas=None, # Let run_isal_process load them for CLI
            persona_sets=None, # Let run_isal_process load them for CLI
            rich_console=console # Pass the main console instance for rich output
        )
        # Then, run the debate process
        final_answer, intermediate_steps = debate_instance.run_debate()
        
        if verbose:
            console.print(Panel(Text("--- Intermediate Steps ---", justify="center"), style="bold magenta"))
            # Filter out Total_Tokens_Used and Total_Estimated_Cost_USD for this display
            step_keys_to_process = [k for k in intermediate_steps.keys() 
                                    if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD"]
            
            for step_key in step_keys_to_process:
                console.print(f"\n[bold cyan]### {step_key.replace('_', ' ').title()}:[/bold cyan]")
                content = intermediate_steps.get(step_key, "N/A")
                # Construct the token count key based on common patterns
                token_count_key = None
                if step_key.endswith("_Output"):
                    token_count_key = step_key.replace("_Output", "_Tokens_Used")
                elif step_key.endswith("_Critique"):
                    token_count_key = step_key.replace("_Critique", "_Tokens_Used")
                elif step_key.endswith("_Feedback"):
                    token_count_key = step_key.replace("_Feedback", "_Tokens_Used")
                else: # For steps like 'Devils_Advocate' without a suffix
                    token_count_key = f"{step_key}_Tokens_Used"
                    
                tokens_used = intermediate_steps.get(token_count_key, "N/A")
                
                console.print(Syntax(content, "markdown", theme="monokai", line_numbers=False, word_wrap=True))
                console.print(f"[bold]Tokens Used for this step:[/bold] {tokens_used}")

    except TokenBudgetExceededError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        # Access from debate_instance
        total_tokens = debate_instance.intermediate_steps.get("Total_Tokens_Used", 0) if debate_instance else 0
        total_cost = debate_instance.intermediate_steps.get("Total_Estimated_Cost_USD", 0.0) if debate_instance else 0.0
        console.print(f"[bold yellow]Process halted due to budget.[/bold yellow]")
        console.print(f"[bold]Final Tokens Used:[/bold] {total_tokens}")
        console.print(f"[bold]Final Estimated Cost:[/bold] {total_cost}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        # Access from debate_instance if available, otherwise default
        if debate_instance is not None:
            total_tokens = debate_instance.intermediate_steps.get("Total_Tokens_Used", 0)
            total_cost = debate_instance.intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)
        else:
            total_tokens = 0
            total_cost = 0.0
        console.print(f"[bold]Final Tokens Used:[/bold] {total_tokens}")
        console.print(f"[bold]Final Estimated Cost:[/bold] {total_cost}")
        raise typer.Exit(code=1)

    console.print(Panel(Text("--- Final Synthesized Answer ---", justify="center"), style="bold green"))
    console.print(Syntax(final_answer, "markdown", theme="monokai", line_numbers=False, word_wrap=True))
    
    total_tokens = intermediate_steps.get("Total_Tokens_Used", 0)
    total_cost = intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)
    console.print(Panel(Text(f"Total Tokens Consumed: [bold]{total_tokens:,}[/bold]\nTotal Estimated Cost: [bold]${total_cost:.4f}[/bold]", justify="center"), style="bold green"))

if __name__ == "__main__":
    app()