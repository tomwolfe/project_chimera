# main.py
import typer
import os
from core import run_isal_process, TokenBudgetExceededError, parse_llm_code_output, validate_code_output, format_git_diff
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from typing import List, Optional

app = typer.Typer(help="Project Chimera: Socratic Self-Debate with LLMs for reasoning and code generation.")
console = Console()

@app.command()
def reason(
    prompt: str = typer.Argument(..., help="The initial prompt for the Socratic Arbitration Loop."),
    context_files: Optional[List[str]] = typer.Option(None, "--context", "-c", help="Path to a relevant file to include as context. Can be used multiple times."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all intermediate reasoning steps."),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Your Gemini API key.", envvar="GEMINI_API_KEY"),
    max_tokens_budget: int = typer.Option(100000, "--max-tokens", "-m", help="Maximum total tokens for the entire process."),
    domain: str = typer.Option("auto", "--domain", "-d", help="Reasoning domain (e.g., General, Software Engineering)."),
    model_name: str = typer.Option("gemini-2.5-flash-lite", "--model", "-M", help="The LLM model to use.")
):
    """
    Runs the Socratic Arbitration Loop on a given prompt, with optional codebase context.
    """
    if not api_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set or --api-key not provided.")
        raise typer.Exit(code=1)

    # Load codebase context from files
    codebase_context = {}
    if context_files:
        if len(context_files) > 3:
            console.print("[bold yellow]Warning:[/bold yellow] More than 3 context files provided. Using the first 3.")
            context_files = context_files[:3]
        for file_path in context_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    codebase_context[file_path] = f.read()
                console.print(f"[green]Loaded context from:[/] {file_path}")
            except FileNotFoundError:
                console.print(f"[bold red]Error:[/bold red] Context file not found: {file_path}")
                raise typer.Exit(code=1)
            except Exception as e:
                console.print(f"[bold red]Error reading file {file_path}:[/bold red] {e}")
                raise typer.Exit(code=1)

    def cli_status_callback(message: str, state: str = "running", expanded: bool = True,
                            current_total_tokens: int = 0, current_total_cost: float = 0.0,
                            estimated_next_step_tokens: int = 0, estimated_next_step_cost: float = 0.0):
        status_color = "blue"
        if state == "error": status_color = "red"
        elif state == "warning": status_color = "yellow"
        elif state == "complete": status_color = "green"

        console.print(f"[{status_color}]Status:[/{status_color}] {message}")
        console.print(f"  [bold]Tokens Used:[/bold] {current_total_tokens} | [bold]Estimated Cost:[/bold] ${current_total_cost:.4f}")
        if estimated_next_step_tokens > 0:
            budget_remaining = max_tokens_budget - current_total_tokens
            if estimated_next_step_tokens > budget_remaining:
                console.print(f"  [bold red]WARNING:[/bold red] Next step ({estimated_next_step_tokens} tokens / ${estimated_next_step_cost:.4f}) will exceed budget ({budget_remaining} remaining).")
            else:
                console.print(f"  [bold yellow]Next Step Estimate:[/bold] {estimated_next_step_tokens} tokens / ${estimated_next_step_cost:.4f} (Budget remaining: {budget_remaining})")
        console.print("-" * 80)

    debate_instance = None
    try:
        console.print(Panel(Text(f"🤖 Starting Socratic Debate (Framework: {domain}, Budget: {max_tokens_budget} tokens)...", justify="center"), style="bold blue"))
        
        debate_instance = run_isal_process(
            prompt=prompt, api_key=api_key, max_total_tokens_budget=max_tokens_budget,
            model_name=model_name, domain=domain,
            streamlit_status_callback=cli_status_callback,
            codebase_context=codebase_context,
            rich_console=console
        )
        final_answer, intermediate_steps = debate_instance.run_debate()
        
        # --- Display Results ---
        console.print(Panel(Text("--- Final Results ---", justify="center"), style="bold green"))
        
        # If Software Engineering, parse and display structured output
        if debate_instance.domain == "Software Engineering":
            parsed_data = parse_llm_code_output(final_answer)
            validation_results = validate_code_output(parsed_data, codebase_context)

            console.print("\n[bold]Commit Message Suggestion:[/bold]")
            console.print(Panel(parsed_data['summary'].get('commit_message', 'N/A'), style="yellow"))
            
            console.print("\n[bold]Rationale:[/bold]")
            console.print(parsed_data['summary'].get('rationale', 'N/A'))

            if parsed_data['summary'].get('conflict_resolution'):
                console.print("\n[bold green]Conflict Resolution:[/bold green]")
                console.print(parsed_data['summary']['conflict_resolution'])
            if parsed_data['summary'].get('unresolved_conflict'):
                console.print("\n[bold yellow]Unresolved Conflict:[/bold yellow]")
                console.print(parsed_data['summary']['unresolved_conflict'])

            console.print("\n[bold]Validation Report:[/bold]")
            if not validation_results['issues'] and not validation_results['malformed_blocks']:
                console.print("[green]✅ No issues detected.[/green]")
            else:
                for issue in validation_results['issues']:
                    console.print(f"[yellow]⚠️ {issue['type']} in `{issue['file']}`:[/yellow] {issue['message']} (Line: {issue.get('line', 'N/A')})")
                if validation_results['malformed_blocks']:
                    console.print(f"[bold red]Malformed Output Detected:[/bold red] The LLM produced {len(validation_results['malformed_blocks'])} block(s) that could not be parsed. Raw output is shown below.")
            
            console.print("\n[bold]Proposed Code Changes:[/bold]")
            for file_path, change in parsed_data['changes'].items():
                console.print(Panel(f"📄 {file_path} ({change['type']})", style="bold cyan"))
                if change['type'] == 'ADD':
                    console.print(Syntax(change['content'], 'python', theme="monokai", line_numbers=True))
                elif change['type'] == 'MODIFY':
                    original_content = codebase_context.get(file_path, "")
                    diff_text = format_git_diff(original_content, change['new_content'])
                    console.print(Syntax(diff_text, 'diff', theme="monokai"))
                elif change['type'] == 'REMOVE':
                    # For REMOVE, the LLM provides lines to remove. Display them with '-' prefix.
                    console.print(Syntax('\n'.join([f"- {line}" for line in change['lines']]), 'diff', theme="monokai"))
            
            # Display malformed blocks as fallbacks
            for block in validation_results['malformed_blocks']:
                console.print(Panel("Unknown File (Malformed Block)", style="bold red"))
                console.print("[red]This block was malformed and could not be parsed correctly. Raw output is shown below.[/red]")
                console.print(Syntax(block, 'text', theme="monokai"))

        else:
            console.print(Syntax(final_answer, "markdown", theme="monokai", word_wrap=True))

        if verbose:
            console.print(Panel(Text("--- Intermediate Steps ---", justify="center"), style="bold magenta"))
            step_keys_to_process = [k for k in intermediate_steps.keys() 
                                    if not k.endswith("_Tokens_Used") and k != "Total_Tokens_Used" and k != "Total_Estimated_Cost_USD"]
            
            for step_key in step_keys_to_process:
                console.print(f"\n[bold cyan]### {step_key.replace('_', ' ').title()}:[/bold cyan]")
                content = intermediate_steps.get(step_key, "N/A")
                token_count_key = f"{step_key.replace('_Output', '').replace('_Critique', '').replace('_Feedback', '')}_Tokens_Used"
                tokens_used = intermediate_steps.get(token_count_key, "N/A")
                
                console.print(Syntax(content, "markdown", theme="monokai", line_numbers=False, word_wrap=True))
                console.print(f"[bold]Tokens Used for this step:[/bold] {tokens_used}")

    except (TokenBudgetExceededError, Exception) as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
        total_tokens = debate_instance.intermediate_steps.get("Total_Tokens_Used", 0) if debate_instance else 0
        total_cost = debate_instance.intermediate_steps.get("Total_Estimated_Cost_USD", 0.0) if debate_instance else 0.0
        console.print(f"[bold]Final Tokens Used:[/bold] {total_tokens}")
        console.print(f"[bold]Final Estimated Cost:[/bold] {total_cost}")
        raise typer.Exit(code=1)

    total_tokens = intermediate_steps.get("Total_Tokens_Used", 0)
    total_cost = intermediate_steps.get("Total_Estimated_Cost_USD", 0.0)
    console.print(Panel(Text(f"Total Tokens: [bold]{total_tokens:,}[/bold] | Est. Cost: [bold]${total_cost:.4f}[/bold]", justify="center"), style="bold green"))

if __name__ == "__main__":
    app()