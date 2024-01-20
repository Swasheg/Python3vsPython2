import requests
from bs4 import BeautifulSoup
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

program_counter = 1

def scrape_data_geeks_for_geeks(url, output_file):
    global program_counter
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Check if it is Python3
        python3_heading = soup.find('h2', class_='tabtitle')
        if python3_heading and 'Python3' in python3_heading.text:
            # Replace 'td-class' with the actual <td> class name you want to scrape
            td_elements = soup.find_all('div', class_='code-container')

            if td_elements:
                with open(output_file, 'a', encoding='utf-8') as file:
                    for td in td_elements:
                        data = td.text.strip()
                        if data:
                            file.write(f"#Program_number_{program_counter}: \n{data}\n\n")
                            program_counter += 1
    else:
        print(f"Failed to fetch data from {url}")


def scrape_data_w3schools(url, output_file):
    global program_counter
    try:
        print(url)

        # Use headless mode (no GUI) to speed up the process
        chrome_options = Options()
        chrome_options.add_argument("--headless")

        # Start the Chrome browser
        driver = webdriver.Chrome(options=chrome_options)

        # Fetch the webpage
        driver.get(url)

        # Wait for the CodeMirror elements to be present
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'CodeMirror-code')))

        # Get the page source after JavaScript execution
        page_source = driver.page_source

        # Parse the HTML content
        soup = BeautifulSoup(page_source, 'html.parser')

        # Your existing code to extract data
        div_elements = soup.find_all('div', class_='CodeMirror-code')
        print(div_elements)
        if div_elements:
            with open(output_file, 'a', encoding='utf-8') as file:
                # Inside scrape_data_w3schools function
                for div in div_elements:
                    pre_elements = div.find_all('pre', class_='CodeMirror-line')
                    current_program = []

                    for pre in pre_elements:
                        # Extract the code with proper indentation
                        code_line = ' '.join(
                            [span.text.replace('\xa0', '') for span in pre.find_all('span', class_='cm-m-python')])

                        if code_line:
                            current_program.append(code_line)

                    if current_program:
                        # Write data with a label "program X:"
                        file.write(f"#Program_number_{program_counter}:\n")
                        file.write('\n'.join(current_program) + '\n\n')
                        program_counter += 1

        # Close the browser
        driver.quit()
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")


def scrape_links_and_data_w3_schools(main_url, output_file):
    response = requests.get(main_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Add conditions for the outer div
        outer_div = soup.find('div', class_='w3-col l10 m12', id='main')

        if outer_div:
            # Find all 'w3-bar-block' elements that are direct children of the outer div
            content_blocks = outer_div.find_all('div', class_='w3-bar-block')

            # Iterate over each 'w3-bar-block' element
            for content in content_blocks:
                # Print the HTML content of 'content' for debugging
                print("Content HTML:")
                print(content)

                # Replace 'link-class' with the actual class name of links you want to click
                link_elements = content.select('a')

                # Print the length of 'link_elements' for debugging
                print("Number of links found:", len(link_elements))

                for link in link_elements:
                    link_url = link.get('href')
                    if link_url:
                        print(link_url)
                        full_link_url = 'https://www.w3schools.com/python/' + link_url if not link_url.startswith('http') else link_url

                        print(f"Processing link: {full_link_url}")  # Print the link being processed

                        # Scrape data from the linked page and save it to the output file
                        scrape_data_w3schools(full_link_url, output_file)

                        print(f"Data from {full_link_url} saved to {output_file}")
    else:
        print(f"Failed to fetch data from {main_url}")



# Main scraping function
def scrape_links_and_data_geeks_for_geeks(main_url, output_file, class_name):
    response = requests.get(main_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', class_=class_name)

        link_elements = content.select('a[href^="https://www.geeksforgeeks.org/"]')

        for link in link_elements:
            link_url = link.get('href')
            if link_url:
                full_link_url = main_url + link_url if not link_url.startswith('http') else link_url

                # Scrape data from the linked page and save it to the output file
                scrape_data_geeks_for_geeks(full_link_url, output_file)

                print(f"Data from {full_link_url} saved to {output_file}")
    else:
        print(f"Failed to fetch data from {main_url}")

# Function to remove comments from a file
def remove_comments_and_blank_lines(input_filepath, output_filepath):
    with open(input_filepath, 'r') as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        # Split the line based on comments
        parts = re.split(r'#', line, maxsplit=1)

        # Keep the first part (code) and preserve leading whitespace
        modified_line = parts[0].rstrip('\n')

        # Add the modified line only if it is not blank
        if modified_line.strip():
            modified_lines.append(modified_line + '\n')

    with open(output_filepath, 'w') as output_file:
        output_file.write(''.join(modified_lines))

# Function to replace [NBSP] characters with regular spaces in a file
def replace_nbsp_characters(file_path):
    # Read the content of the file
    with open(file_path, 'r') as f:
        text = f.read()

    # Replace [NBSP] characters with regular spaces
    text_with_spaces = re.sub(u'\xa0', ' ', text)

    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(text_with_spaces)

def replace_zwsp_characters(file_path):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Replace [ZWSP] characters with an empty string
    text_without_zwsp = re.sub('\u200b', '', text)

    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text_without_zwsp)

if __name__ == "__main__":
    geeks_for_geeks = 'https://www.geeksforgeeks.org/python-programming-examples/'
    w3schools = "https://www.w3schools.com/python/python_examples.asp"
    output_file = 'scraped_data_new_w3_geeks.txt'
    output_file_cleaned = 'scraped_data_new_cleaned.py'
    with open(output_file, 'w', encoding='utf-8') as file:
        pass

    categories = ["simple", "array", "list", "string", "matrix", "dictionary", "tuple", "searchingandsorting",
                  "pattern", "datetime"]

    for category in categories:
        scrape_links_and_data_geeks_for_geeks(geeks_for_geeks, output_file, category)
    scrape_links_and_data_w3_schools(w3schools,output_file)
    remove_comments_and_blank_lines(output_file, output_file_cleaned)
    replace_zwsp_characters(output_file_cleaned)
