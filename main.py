# main.py
import typer
import os
# Corrected imports:
# - run_isal_process is defined in this file (main.py), so it shouldn't be imported from core.
# - TokenBudgetExceededError is still in core.py.
# - parse_llm_code_output, validate_code_output, format_git_diff are now in utils.py.
from core import TokenBudgetExceededError
from utils import parse_llm_code_output, validate_code_output, format_git_diff
from rich.console import Console
from rich.panel import Panel
from rich.text import Text # Ensure Text is imported
from rich.syntax import Syntax
from typing import List, Optional, Callable, Dict # Import Dict here

# --- SocraticDebate related imports (needed for type hinting in run_isal_process) ---
# These should be imported from core.py
from core import SocraticDebate, Persona, FullPersonaConfig, GeminiAPIError, LLMProviderError, LLMUnexpectedError, GeminiProvider
# Also need GeminiProvider for type hinting in run_isal_process, and Callable for the callback, and Dict for type hints.
from llm_provider import GeminiProvider # Ensure this is imported if not already covered by core


app = typer.Typer(help="Project Chimera: Socratic Self-Debate with LLMs for reasoning and code generation.")
console = Console()

# --- run_isal_process function definition ---
# This function is defined here in main.py and is the entry point for both CLI and used by app.py
def run_isal_process(
    prompt: str,
    api_key: str,
    max_total_tokens_budget: int = 10000,
    model_name: str = "gemini-2.5-flash-lite",
    domain: str = "auto",
    streamlit_status_callback: Callable = None,
    all_personas: Optional[Dict[str, Persona]] = None,
    persona_sets: Optional[Dict[str, List[str]]] = None,
    personas_override: Optional[Dict[str, Persona]] = None,
    gemini_provider: Optional[GeminiProvider] = None,
    rich_console: Optional[Console] = None,
    codebase_context: Optional[Dict[str, str]] = None
) -> 'SocraticDebate':
    """Initializes and returns the SocraticDebate instance."""
    
    # Load personas and sets if not provided (for CLI or initial app load)
    if all_personas is None or persona_sets is None:
        all_personas, persona_sets, default_set = core.load_personas()
    else:
        # If provided, determine default set from the provided persona_sets
        default_set = "General" if "General" in persona_sets else next(iter(persona_sets.keys()))

    # Determine domain to use
    if domain == "auto" and prompt.strip() and api_key.strip(): # Only auto-recommend if prompt and key are present
        try:
            # Use the provided gemini_provider or create one if not available
            provider_for_domain_rec = gemini_provider or GeminiProvider(api_key=api_key, model_name=model_name)
            llm_recommended_domain = provider_for_domain_rec.recommend_domain(prompt) # Call the method on the provider
            
            if llm_recommended_domain in persona_sets:
                domain = llm_recommended_domain
            else:
                domain = default_set
        except Exception as e:
            # Use rich_console if available, otherwise fallback to print
            if rich_console:
                rich_console.print(f"[yellow]Error during domain recommendation: {e}. Falling back to default domain.[/yellow]")
            else:
                print(f"[yellow]Error during domain recommendation: {e}. Falling back to default domain.[/yellow]")
            domain = default_set
            
    elif domain not in persona_sets:
        domain = default_set
    
    # Get the personas for the selected domain
    if personas_override:
        personas = personas_override
        # If overriding, domain is effectively custom, but we might still want to use the selected domain name for logging/context
        # For simplicity, we'll just use the provided personas and keep the domain name as is or set to 'Custom'
        domain = domain if domain != "auto" else "Custom" # Ensure domain is set if auto was used
    else:
        personas = {name: all_personas[name] for name in persona_sets[domain]}

    # Prepare kwargs for SocraticDebate.__init__
    kwargs_for_debate = {
        'initial_prompt': prompt,
        'api_key': api_key,
        'max_total_tokens_budget': max_total_tokens_budget,
        'model_name': model_name,
        'personas': personas,
        'all_personas': all_personas,
        'persona_sets': persona_sets,
        'domain': domain,
        'status_callback': streamlit_status_callback,
        'rich_console': rich_console,
        'codebase_context': codebase_context,
        'gemini_provider': gemini_provider
    }
    
    debate = SocraticDebate(**kwargs_for_debate)
    return debate

@app.command()
def reason(
    prompt: str = typer.Argument(..., help="The initial prompt for the Socratic Arbitration Loop."),
    context_files: Optional[List[str]] = typer.Option(None, "--context", "-c", help="Path to a relevant file to include as context. Can be used multiple times. Limit of 25 files applies."),
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
        # --- MODIFICATION START ---
        # Increased the limit for context files from 3 to 25
        if len(context_files) > 25:
            console.print("[bold yellow]Warning:[/bold yellow] More than 25 context files provided. Using the first 25.")
            context_files = context_files[:25]
        # --- MODIFICATION END ---
        for file_path in context_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    codebase_context[file_path] = f.read()
                console.print(f"[green]Loaded context from:[/green] {file_path}")
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

        console.print(f"[{status_color}]Status:[/color] {message}")
        console.print(f"  [bold]Tokens Used:[/bold] {current_total_tokens} | [bold]Estimated Cost:[/bold] ${current_total_cost:.4f}")
        if estimated_next_step_tokens > 0:
            budget_remaining = max_tokens_budget - current_total_tokens
            if estimated_next_step_tokens > budget_remaining:
                console.print(f"  [bold red]WARNING:[/bold red] Next step ({estimated_next_step_tokens} tokens / ${estimated_next_step_cost:.4f}) will exceed budget ({budget_remaining} remaining).")
            else:
                console.print(f"  [bold yellow]Next Step Estimate:[/bold yellow] {estimated_next_step_tokens} tokens / ${estimated_next_step_cost:.4f} (Budget remaining: {budget_remaining})")
        console.print("-" * 80)

    debate_instance = None
    try:
        # Ensure the Text object receives a single string with embedded newlines if needed, not literal \n characters
        debate_start_message = f"🤖 Starting Socratic Debate (Framework: {domain}, Budget: {max_tokens_budget} tokens)..."
        console.print(Panel(Text(debate_start_message, justify="center"), style="bold blue"))
        
        # Call the run_isal_process function defined in this file
        debate_instance = run_isal_process(
            prompt=prompt, api_key=api_key, max_total_tokens_budget=max_tokens_budget,
            model_name=model_name, domain=domain,
            streamlit_status_callback=cli_status_callback,
            codebase_context=codebase_context,
            rich_console=console
        )
        final_answer, intermediate_steps = debate_instance.run_debate()
        
        # --- Display Results ---
        console.print(Panel(Text("--- Final Results ---"), justify="center"), style="bold green")
        
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
            console.print(Panel(Text("--- Intermediate Steps ---"), justify="center"), style="bold magenta")
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
    # Corrected line: Removed the extra closing parenthesis
    console.print(Panel(Text(f"Total Tokens: [bold]{total_tokens:,}[/bold] | Est. Cost: [bold]${total_cost:.4f}[/bold]"), justify="center"), style="bold green")

if __name__ == "__main__":
    app()