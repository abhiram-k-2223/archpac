from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import os
import subprocess
from colorama import Fore, Style, init
from tqdm import tqdm

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Color constants
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RED = Fore.RED
MAGENTA = Fore.MAGENTA
BLUE = Fore.BLUE
BRIGHT = Style.BRIGHT
RESET = Style.RESET_ALL

# Load model and data
def load_resources():
    print(f"{YELLOW}Loading embeddings model and package data...{RESET}")

    with tqdm(total=2, desc="Loading resources", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        pbar.update(1)

        try:
            with open('packages_with_clusters.json', 'r') as file:
                packages = json.load(file)
            pbar.update(1)
        except FileNotFoundError:
            print(f"{RED}Error: packages_with_clusters.json not found!{RESET}")
            print(f"Make sure the file exists in the same directory as this script.")
            sys.exit(1)

    return model, packages

def title_split(s):
    return s.split(" ")[0]

def find_similar_packages(query, packages, model, top_k=20):
    query_embedding = model.encode(query)

    # Show loading animation
    print(f"{YELLOW}Searching for packages similar to {CYAN}{query}{YELLOW}...{RESET}")
    sys.stdout.flush()

    with tqdm(total=len(packages), desc="Processing packages", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        similarities = []
        for pkg in packages:
            similarities.append(
                (pkg['title'], cosine_similarity([pkg['embedding']], [query_embedding])[0][0])
            )
            pbar.update(1)

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_logo():
    try:
        with open("logo", "r") as file:
            logo = file.read()
        print(f"{CYAN}{logo}{RESET}")
    except FileNotFoundError:
        # Fallback ASCII art logo if the file doesn't exist
        print(f"""{CYAN}
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║   {MAGENTA}█████{CYAN}╗ {MAGENTA}██████{CYAN}╗  {MAGENTA}██████{CYAN}╗{MAGENTA}██{CYAN}╗  {MAGENTA}██{CYAN}╗{MAGENTA}██████{CYAN}╗  {MAGENTA}█████{CYAN}╗  {MAGENTA}██████{CYAN}╗ ║
    ║  {MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}╗{MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}╗{MAGENTA}██{CYAN}╔════╝{MAGENTA}██{CYAN}║  {MAGENTA}██{CYAN}║{MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}╗{MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}╗{MAGENTA}██{CYAN}╔════╝ ║
    ║  {MAGENTA}███████{CYAN}║{MAGENTA}██████{CYAN}╔╝{MAGENTA}██{CYAN}║     {MAGENTA}███████{CYAN}║{MAGENTA}██████{CYAN}╔╝{MAGENTA}███████{CYAN}║{MAGENTA}██{CYAN}║      ║
    ║  {MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}║{MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}╗{MAGENTA}██{CYAN}║     {MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}║{MAGENTA}██{CYAN}╔═══╝ {MAGENTA}██{CYAN}╔══{MAGENTA}██{CYAN}║{MAGENTA}██{CYAN}║      ║
    ║  {MAGENTA}██{CYAN}║  {MAGENTA}██{CYAN}║{MAGENTA}██{CYAN}║  {MAGENTA}██{CYAN}║╚{MAGENTA}██████{CYAN}╗{MAGENTA}██{CYAN}║  {MAGENTA}██{CYAN}║{MAGENTA}██{CYAN}║     {MAGENTA}██{CYAN}║  {MAGENTA}██{CYAN}║╚{MAGENTA}██████{CYAN}╗ ║
    ║  {RESET}╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝{CYAN} ║
    ║                                           ║
    ║     {YELLOW}Smart Package Search for Arch Linux{CYAN}     ║
    ╚═══════════════════════════════════════════╝{RESET}
        """)

def display_help():
    print(f"\n{BRIGHT}{YELLOW}Available commands:{RESET}")
    print(f"  {CYAN}search [query]{RESET} - Search for packages")
    print(f"  {CYAN}help{RESET} - Show this help message")
    print(f"  {CYAN}clear{RESET} - Clear the screen")
    print(f"  {CYAN}exit{RESET} - Exit the program")
    print()

def display_package_details(package):
    clear_screen()

    description = package.get("Description:", "No description available.")
    repo = package.get("Repository:", "Unknown repository")
    maintainer = package.get("Maintainers:", "Unknown maintainer")
    package_name = title_split(package["title"])

    print(f"\n{BRIGHT}{MAGENTA}╔{'═' * 60}╗{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{' ' * 60}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{YELLOW}  PACKAGE DETAILS: {CYAN}{package['title']}{' ' * (36 - len(package['title']))}{MAGENTA}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{' ' * 60}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}╠{'═' * 60}╣{RESET}")

    # Description with proper wrapping
    print(f"{BRIGHT}{MAGENTA}║{RESET} {GREEN}Description:{RESET}")
    wrapped_desc = [description[i:i+56] for i in range(0, len(description), 56)]
    for line in wrapped_desc:
        print(f"{BRIGHT}{MAGENTA}║{RESET}   {line}{' ' * (57 - len(line))}{BRIGHT}{MAGENTA}║{RESET}")

    print(f"{BRIGHT}{MAGENTA}║{' ' * 60}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{RESET} {GREEN}Repository:{RESET} {repo}{' ' * (48 - len(repo))}{BRIGHT}{MAGENTA}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{RESET} {GREEN}Maintainer:{RESET} {maintainer}{' ' * (47 - len(maintainer))}{BRIGHT}{MAGENTA}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{' ' * 60}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{RESET} {GREEN}Installation:{RESET}{' ' * 47}{BRIGHT}{MAGENTA}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{RESET}   {CYAN}sudo pacman -S {package_name}{' ' * (41 - len(package_name))}{BRIGHT}{MAGENTA}║{RESET}")
    print(f"{BRIGHT}{MAGENTA}║{' ' * 60}║{RESET}")

    # Additional package information if available
    for key, value in package.items():
        if key not in ['title', 'embedding', 'Description:', 'Repository:', 'Maintainers:'] and isinstance(value, str):
            if len(key) <= 55:  # Only display if the key is not too long
                print(f"{BRIGHT}{MAGENTA}║{RESET} {GREEN}{key}{RESET} {value}{' ' * (59 - len(key) - len(value))}{BRIGHT}{MAGENTA}║{RESET}")

    print(f"{BRIGHT}{MAGENTA}╚{'═' * 60}╝{RESET}")

    print(f"\n{YELLOW}Actions:{RESET}")
    print(f"  [1] {CYAN}Install package{RESET}")
    print(f"  [2] {CYAN}Return to results{RESET}")
    print(f"  [3] {CYAN}New search{RESET}")

    while True:
        choice = input(f"\n{GREEN}Choose an option (1-3):{RESET} ")
        if choice == '1':
            print(f"\n{YELLOW}Would run:{RESET} sudo pacman -S {package_name}")
            print(f"{YELLOW}In a real implementation, this would execute the package installation.{RESET}")
            input(f"\n{CYAN}Press Enter to continue...{RESET}")
            return
        elif choice == '2':
            return
        elif choice == '3':
            return 'new_search'

def paginate_results(results, packages, page_size=5):
    total_pages = (len(results) + page_size - 1) // page_size  # Ceiling division
    current_page = 1
    selected_package = None

    while True:
        clear_screen()
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(results))
        page_results = results[start_idx:end_idx]

        display_logo()

        # Display results for current page
        print(f"\n{BRIGHT}{GREEN}Search Results (Page {current_page}/{total_pages}):{RESET}\n")

        for idx, (package_name, score) in enumerate(page_results, start=start_idx + 1):
            package = next(pkg for pkg in packages if pkg["title"] == package_name)
            description = package.get("Description:", "No description available.")
            # Truncate description if too long
            if len(description) > 60:
                description = description[:57] + "..."

            print(f"{CYAN}[{idx}]{RESET} {BRIGHT}{YELLOW}{package_name}{RESET}")
            print(f"    {description}")
            print(f"    {MAGENTA}Similarity:{RESET} {score:.2f}\n")

        # Navigation options
        print(f"\n{YELLOW}Navigation:{RESET}")
        nav_options = []

        if current_page > 1:
            nav_options.append(f"[p] {CYAN}Previous page{RESET}")

        if current_page < total_pages:
            nav_options.append(f"[n] {CYAN}Next page{RESET}")

        nav_options.append(f"[#] {CYAN}Select package by number{RESET}")
        nav_options.append(f"[s] {CYAN}New search{RESET}")
        nav_options.append(f"[q] {CYAN}Back to main menu{RESET}")

        print("  " + "  ".join(nav_options))

        # Get user choice
        choice = input(f"\n{GREEN}Enter your choice:{RESET} ").lower()

        if choice == 'p' and current_page > 1:
            current_page -= 1
        elif choice == 'n' and current_page < total_pages:
            current_page += 1
        elif choice == 'q':
            break
        elif choice == 's':
            return 'new_search'
        elif choice.isdigit():
            pkg_idx = int(choice)
            if 1 <= pkg_idx <= len(results):
                package_name = results[pkg_idx-1][0]
                selected_package = next(pkg for pkg in packages if pkg["title"] == package_name)
                action = display_package_details(selected_package)
                if action == 'new_search':
                    return 'new_search'
            else:
                print(f"{RED}Invalid package number. Please try again.{RESET}")
                time.sleep(1)
        else:
            print(f"{RED}Invalid choice. Please try again.{RESET}")
            time.sleep(1)

def search_packages(query, packages, model):
    results = find_similar_packages(query, packages, model)
    return paginate_results(results, packages)

def main():
    model, packages = load_resources()

    while True:
        clear_screen()
        display_logo()
        display_help()

        try:
            cmd = input(f"{BRIGHT}{GREEN}archpac>{RESET} ").strip()

            if cmd.lower() == 'exit' or cmd.lower() == 'quit':
                print(f"\n{YELLOW}Thank you for using ArchPac! Goodbye!{RESET}")
                break

            elif cmd.lower() == 'clear':
                continue  # Screen will be cleared at the start of the loop

            elif cmd.lower() == 'help':
                input(f"\n{CYAN}Press Enter to continue...{RESET}")

            elif cmd.lower().startswith('search ') or not cmd.lower().startswith(('exit', 'quit', 'clear', 'help')):
                # If command starts with 'search', use the rest as query
                # If it doesn't start with any known command, treat the whole input as a search query
                if cmd.lower().startswith('search '):
                    query = cmd[7:].strip()
                else:
                    query = cmd

                if not query:
                    print(f"{RED}Please provide a search query.{RESET}")
                    time.sleep(1)
                    continue

                action = search_packages(query, packages, model)
                if action != 'new_search':
                    input(f"\n{CYAN}Press Enter to return to main menu...{RESET}")

            else:
                print(f"{RED}Unknown command. Type 'help' for a list of commands.{RESET}")
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Interrupted. Press Ctrl+C again to exit or Enter to continue.{RESET}")
            try:
                if input() == '':
                    continue
                else:
                    print(f"\n{YELLOW}Exiting ArchPac. Goodbye!{RESET}")
                    break
            except KeyboardInterrupt:
                print(f"\n{YELLOW}Exiting ArchPac. Goodbye!{RESET}")
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}ArchPac Exiting...{RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"{RED}An unexpected error occurred: {str(e)}{RESET}")
        sys.exit(1)
