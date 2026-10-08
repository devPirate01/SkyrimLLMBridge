"""
memory_wipe.py

Utility script to completely erase all NPC conversation memory
and session logs from the Skyrim LLM Bridge.

Usage:
    python memory_wipe.py
"""

import os
import shutil
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BRIDGE_DIR / "memory"
LOGS_DIR = BRIDGE_DIR / "logs"

def wipe_directory(directory: Path):
    if not directory.exists():
        print(f"Directory {directory.name} does not exist. Skipping.")
        return

    count = 0
    for item in directory.iterdir():
        if item.is_file():
            try:
                item.unlink()
                count += 1
            except Exception as e:
                print(f"Failed to delete {item.name}: {e}")
    
    print(f"Wiped {count} files from {directory.name}/")

def main():
    print("WARNING: This will permanently delete all NPC conversation memory and logs.")
    confirm = input("Type 'yes' to proceed: ")
    
    if confirm.lower().strip() != 'yes':
        print("Aborted.")
        return

    print("\nWiping Memory...")
    wipe_directory(MEMORY_DIR)
    
    print("\nWiping Logs...")
    wipe_directory(LOGS_DIR)

    print("\nMemory wipe complete! You are ready to start a fresh playthrough.")

if __name__ == "__main__":
    main()
