import os
import argparse
import shutil
import glob

def read_gitignore():
    gitignore = set()
    if os.path.isfile('.gitignore'):
        with open('.gitignore', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    gitignore.add(line)
    return gitignore


def generate_code(foldername, output_directory):
    if foldername.lower() == 'reinforcement-learning':
        folder_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    else:
        folder_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, foldername)

    print(f'Folder directory: {folder_directory}')
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory, exist_ok=True)

    code_found = False
    all_code = ''
    for root, dirs, files in os.walk(folder_directory):
        # Exclude directories listed in the .gitignore file and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['.git', 'venv']]
        
        print(f'Processing directory: {root}')
        for filename in files:
            if filename.endswith('.py'):
                print(f'Python file found: {filename}')
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    code = f.read()
                    all_code += code + '\n'
                    code_found = True
    if code_found:
        print(f'Input for GPT generated.')
        all_code_file = os.path.join(output_directory, f'{foldername}_code.txt')
        with open(all_code_file, 'w') as out_f:
            out_f.write(all_code)
    else:
        print('Error: No code found.')
    os.startfile(output_directory)

def generate_structure(foldername, output_file):
    folder_directory = os.path.join(os.getcwd(), os.pardir, foldername)
    gitignore = read_gitignore()
    with open(output_file, 'w') as f:
        f.write(foldername + '\n')
        for root, dirs, files in os.walk(folder_directory):
            # Exclude directories listed in the .gitignore file and env
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['.git', '__pycache__', 'env']]
            files = [f for f in files if not f.startswith('.') and f.endswith('.py')]
            indent_level = root.count(os.sep) - folder_directory.count(os.sep)
            if indent_level == 0:
                continue
            f.write('{}{}\n'.format('    ' * indent_level, os.path.basename(root)))
            subindent = '    ' * (indent_level + 1)
            for file in files:
                f.write('{}{}\n'.format(subindent, file))

    print(f'File {output_file} created.')

                    
def generate_all(foldername, output_directory):
    root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    output_directory = os.path.join(root_directory,  'gpt')
    folder_directory = os.path.join(output_directory, foldername)
    os.makedirs(folder_directory, exist_ok=True)

    # generate code
    generate_code(foldername, folder_directory)

    # generate structure
    structure_output_file = os.path.join(folder_directory, f'{foldername}_structure.txt')
    generate_structure(foldername, structure_output_file)



    # generate other files
    current_file_num = 1
    functions = {k: v for k, v in globals().items() if callable(v) and k.startswith('generate_') and len(v.__code__.co_varnames) == 2 and v.__code__.co_varnames[0] == 'foldername' and v.__code__.co_varnames[1] == 'output_directory'}
    for name, func in functions.items():
        func_name = name[len('generate_'):]
        if func_name == 'structure' or func_name == 'code':
            continue
        output_file = os.path.join(folder_directory, f'{foldername}_{func_name}_{current_file_num}.txt')
        func(foldername, output_file)
        print(f'File {output_file} created.')
        current_file_num += 1

    # merge code files
    all_code_file = os.path.join(folder_directory, f'{foldername}_code.txt')
    with open(all_code_file, 'w') as out_f:
        code_files = glob.glob(os.path.join(folder_directory, f'{foldername}_code*.txt'))
        code_files.sort()
        for file in code_files:
            with open(file, 'r') as in_f:
                out_f.write(in_f.read() + '\n')
    print(f'File {all_code_file} created.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help='the command to run: code, structure, or all', default='all', nargs='?')
    parser.add_argument('foldername', type=str, help='the name of the folder containing the Python files to include in the generated file', default='reinforcement-learning', nargs='?')
    args = parser.parse_args()

    root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    output_directory = os.path.join(root_directory, 'gpt', args.foldername)
    # Fix output_directory path
    output_directory = os.path.abspath(os.path.join(output_directory, os.pardir))

    if args.command == 'code':
        generate_code(args.foldername, output_directory)
    elif args.command == 'structure':
        os.makedirs(output_directory, exist_ok=True)
        generate_structure(args.foldername, output_directory)
    elif args.command == 'all':
        generate_all(args.foldername, output_directory)
    else:
        print(f'Error: Invalid command {args.command}. Use "code", "structure", or "all".')

    # Remove the generated venv folder
    shutil.rmtree(os.path.join(output_directory, 'env'), ignore_errors=True)