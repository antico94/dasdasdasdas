import os

# Folders to ignore
IGNORE_DIRS = {'.venv', '__pycache__', '.git', '.idea', 'print_project_structure.py'}


def build_tree(start_path='.', indent=''):
    """Recursively builds the directory structure as a list of strings, ignoring specified folders."""
    lines = []
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        lines.append(indent + '└── [Permission Denied]')
        return lines

    items = [item for item in items if item not in IGNORE_DIRS]

    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = i == len(items) - 1
        connector = '└── ' if is_last else '├── '
        lines.append(indent + connector + item)
        if os.path.isdir(path):
            extension = '    ' if is_last else '│   '
            lines.extend(build_tree(path, indent + extension))
    return lines


def save_project_structure(root_path='.'):
    """Builds the project structure and saves it to 'project_structure.txt' in the root directory."""
    abs_root = os.path.abspath(root_path)
    header = f"Project structure of: {abs_root}\n\n"
    tree_lines = build_tree(root_path)
    content = header + "\n".join(tree_lines)

    output_file = os.path.join(abs_root, "project_structure.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Project structure saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Save project directory structure to project_structure.txt.')
    parser.add_argument('path', nargs='?', default='.', help='Path to the root directory')
    args = parser.parse_args()

    save_project_structure(args.path)
