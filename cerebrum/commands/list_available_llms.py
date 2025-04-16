from cerebrum.llm.apis import list_available_llms

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED

import sys

def list_agenthub_agents():
    console = Console()
    
    with console.status("[bold green]Listing available LLMs..."):
        llms = list_available_llms()
    
    if not llms:
        console.print(Panel("[bold yellow]No LLMs found", title="LLM List"))
        return
    
    # Create a table with row separators and rounded borders
    table = Table(
        title="Available Agents in AgentHub",
        box=ROUNDED,
        show_header=True,
        header_style="bold white on blue",
        show_lines=True,  # This adds horizontal lines between rows
    )
    
    # Add columns to the table with adjusted widths
    table.add_column("Name", style="cyan bold", no_wrap=True)
    table.add_column("Backend", style="green", width=40, overflow="fold")
    table.add_column("Hostname", style="blue", no_wrap=True)
    
    # Add rows to the table
    for llm in llms:
        name = llm.get("name", "N/A")
        backend = llm.get("backend", "N/A")
        hostname = llm.get("hostname", "N/A")
        
        table.add_row(name, backend, hostname)
    
    # Print the table
    console.print("\n")  # Add some space before the table
    console.print(table)
    
    # Print summary
    summary = Text()
    summary.append(f"\nTotal LLMs available: ", style="bold")
    summary.append(f"{len(llms)}", style="bold green")
    console.print(summary)
    console.print("\n")  # Add some space after the summary

def main():
    list_agenthub_agents()

if __name__ == "__main__":
    sys.exit(main())