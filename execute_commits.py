#!/usr/bin/env python3
"""
Execute simulated git commit history for AI Academy.
Reads commit_plan.json and creates backdated commits.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

# Configuration
SOURCE_DIR = Path("/Users/bharatsamudrala/Desktop/vscode_base/academy")
TARGET_DIR = Path("/Users/bharatsamudrala/Desktop/vscode_base/academy-commits")
COMMIT_PLAN = SOURCE_DIR / "commit_plan.json"

# Author info
AUTHOR_NAME = "Bharat Samudrala"
AUTHOR_EMAIL = "bharat@example.com"  # Update with your email

def run_git(cmd, cwd):
    """Run a git command in the specified directory."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
    return result.returncode == 0

def copy_file(src_path, dest_path):
    """Copy a file or directory, creating parent directories if needed."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_dir():
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)
    else:
        shutil.copy2(src_path, dest_path)

def resolve_files(file_patterns, source_dir):
    """Resolve file patterns to actual file paths."""
    files = []
    for pattern in file_patterns:
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            dir_path = source_dir / pattern.rstrip('/')
            if dir_path.exists() and dir_path.is_dir():
                # Add the whole directory
                files.append(pattern.rstrip('/'))
        else:
            # Regular file
            file_path = source_dir / pattern
            if file_path.exists():
                files.append(pattern)
            else:
                print(f"  Warning: File not found: {pattern}")
    return files

def main():
    print("=" * 60)
    print("AI Academy Commit Execution")
    print("=" * 60)

    # Load commit plan
    with open(COMMIT_PLAN) as f:
        plan = json.load(f)

    commits = plan["commits"]
    print(f"Loaded {len(commits)} commits from plan")

    # Create target directory
    if TARGET_DIR.exists():
        print(f"\nRemoving existing target directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True)
    print(f"Created target directory: {TARGET_DIR}")

    # Initialize git repo
    print("\nInitializing git repository...")
    run_git(["git", "init"], TARGET_DIR)
    run_git(["git", "config", "user.name", AUTHOR_NAME], TARGET_DIR)
    run_git(["git", "config", "user.email", AUTHOR_EMAIL], TARGET_DIR)

    # Process each commit
    print("\n" + "=" * 60)
    print("Processing commits...")
    print("=" * 60)

    for commit in commits:
        commit_id = commit["id"]
        date = commit["date"]
        message = commit["message"]
        files = commit["files"]

        print(f"\n[{commit_id:2d}] {date} | {message[:50]}...")

        # Resolve and copy files
        resolved_files = resolve_files(files, SOURCE_DIR)

        for file_pattern in resolved_files:
            src = SOURCE_DIR / file_pattern
            dest = TARGET_DIR / file_pattern

            try:
                copy_file(src, dest)
                print(f"     + {file_pattern}")
            except Exception as e:
                print(f"     ! Error copying {file_pattern}: {e}")

        # Stage all files
        run_git(["git", "add", "-A"], TARGET_DIR)

        # Create backdated commit
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date

        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=TARGET_DIR,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                print("     (no changes to commit)")
            else:
                print(f"     Error: {result.stderr}")
        else:
            print("     Committed successfully")

    # Summary
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)

    # Show git log
    print("\nRecent commits:")
    result = subprocess.run(
        ["git", "log", "--oneline", "-15"],
        cwd=TARGET_DIR,
        capture_output=True,
        text=True
    )
    print(result.stdout)

    print(f"\nRepository created at: {TARGET_DIR}")
    print("\nTo push to GitHub:")
    print(f"  cd {TARGET_DIR}")
    print("  git remote add origin <your-github-url>")
    print("  git branch -M main")
    print("  git push -u origin main")

if __name__ == "__main__":
    main()
