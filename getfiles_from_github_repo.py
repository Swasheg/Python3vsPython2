import os
import requests
from git import Repo
import shutil
import chardet
import re

def search_github_repositories(query, repo_type='repositories', result_limit=10):
    url = f'https://api.github.com/search/{repo_type}?q={query}&per_page={result_limit}'
    print(url)
    response = requests.get(url, headers={'Accept': 'application/vnd.github.v3+json'})
    
    if response.status_code == 200:
        data = response.json()
        repositories = data.get('items', [])
        return repositories
    else:
        print(f"Error: {response.status_code}")
        return None
    
def has_match_case(source_code):
    return 'match' in source_code and 'case' in source_code

def download_python_files_from_repositories(repositories, destination_folder, output_file, max_lines=100000):
    line_count = 0
    with open(output_file, 'w', encoding='utf-8') as output:
        for repo in repositories:
            if line_count >= max_lines:
                break

            repo_url = repo['clone_url']
            repo_name = repo['name']
            repo_destination = os.path.join(destination_folder, repo_name)

            # Create a new folder if it doesn't exist
            os.makedirs(repo_destination, exist_ok=True)

            # Clone the repository
            Repo.clone_from(repo_url, repo_destination)

            for root, dirs, files in os.walk(repo_destination):
                for file in files:
                    if line_count >= max_lines:
                        break

                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'rb') as py_file:
                                # Detect the encoding of the file
                                result = chardet.detect(py_file.read())
                                encoding = result['encoding']

                            # Read the file using the detected encoding
                            with open(file_path, 'r', encoding=encoding, errors='replace') as py_file:
                                source_code = py_file.read()

                                if has_match_case(source_code):
                                    print(f"Skipping file with both 'match' and 'case' constructs: {file_path}")
                                    continue
                                lines = source_code.split('\n')
                                for line in lines:
                                    if line.strip():  # Skip writing blank lines
                                        output.write(line)
                                        output.write("\n")
                                        line_count += 1
                                        line_count
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
                            continue

def convert_to_python2(input_file):
    # Use 3to2 tool to convert Python 3 code to Python 2
    os.system(f"3to2 -w {input_file}")

def modify_print_statements(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    content = re.sub(r'\*args', 'args', content)
    content = re.sub(r'\**kwargs', 'kwargs', content)
    pattern = re.compile(r'print\s*\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)', re.DOTALL)
    content = re.sub(r'(def\s+[^\s(]+\s*\([^)]*),\s*/([^)]*\)):', r'\1\2:', content)


    matches = list(pattern.finditer(content))

    for match in reversed(matches):
        matched_string = match.group(0)
        replacement_string = re.sub(r',\s*(file|end|flush)\s*=[^,)]*', '', matched_string)
        content = content[:match.start()] +replacement_string+ content[match.end():]
    with open(file_path, 'w') as file:
        file.write(content)


# Set your desired paths
destination_folder = '/Users/hegdeswastik/webscraping_python/python2vspython3/python_script_from_repos_test'
output_file_python3 = '/Users/hegdeswastik/webscraping_python/python2vspython3/python31_script_from_repos_test.py'
output_folder_python2 = '/Users/hegdeswastik/webscraping_python/python2vspython3/python21_script_from_repos_test.py'


search_query = 'python programs'
result_limit = 5

# Search for repositories
max_lines = 50000

# Search for repositories
repositories = search_github_repositories(search_query, result_limit=result_limit)

# Download .py files from the repositories
download_python_files_from_repositories(repositories, destination_folder, output_file_python3, max_lines=max_lines)
print(f"Content of .py files from GitHub repositories has been written to '{output_file_python3}'.")
modify_print_statements(output_file_python3)

shutil.copy2(output_file_python3, output_folder_python2)
print(f"File copied successfully from {output_file_python3} to {output_folder_python2}")

convert_to_python2(output_folder_python2)
print(f"Python 2 format of files has been written to '{output_folder_python2}'.")
