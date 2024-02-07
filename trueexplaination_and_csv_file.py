import pandas as pd
import re
def generate_output_csv(file1_path, file2_path, file3_path, output_file_path):
    # Read the content from the input files
    with open(file1_path, 'r') as file1:
        lines1 = file1.readlines()

    with open(file2_path, 'r') as file2:
        lines2 = file2.readlines()

    with open(file3_path, 'r') as file3:
        lines3 = file3.readlines()


    # Create DataFrames from the lines
    df1 = pd.DataFrame({'FileContent': lines1, 'Version': 1})
    df2 = pd.DataFrame({'FileContent': lines2, 'Version': 0})
    df3 = pd.DataFrame({'FileContent': lines3, 'Version': 1})

    # Concatenate the DataFrames
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    output_df = combined_df.copy()
    # Initialize columns with empty values
    combined_df['true class'] = ""
    combined_df['true explanation'] = ""

    # Fill in the values for the specified columns
    output_df['line of code'] = combined_df['FileContent']
    output_df['true class'] = combined_df['Version']

    patterns = {
        "print(": re.compile(r"print\s*\(\s*(\S+)"),
        "__future__": re.compile(r"(__future__\s+)"),
        "xrange": re.compile(r"(xrange\s*\()"),
        " range": re.compile(r"(\srange\s*\()"),
        'u"': re.compile(r'(u\"\s*(\S+))'),
        "u'": re.compile(r"(u\'\s*(\S+))"),
        "print ": re.compile(r"print\s*\s*(\S+)"),
        "unicode(": re.compile(r"\bunicode\("),
        "__next__(": re.compile(r"\s__next__\s*\("),
        "raw_input(": re.compile(r"\sraw_input\s*\(")
    }

    # Function to extract matched strings for a pattern
    def extract_matches(pattern, text):
        match = re.search(pattern, text)
        return match.group() if match else None

    # Apply the function to each row of the DataFrame
    matched_strings = {key: combined_df['FileContent'].apply(lambda x: extract_matches(pattern, x)) for key, pattern in
                       patterns.items()}

    # Fill in 'true explanation' with the matched strings
    output_df['true explanation'] = ""

    # Fill in 'true explanation' with the matched strings
    output_df['true explanation'] = matched_strings["print("].combine_first(
        matched_strings["__future__"]).combine_first(
        matched_strings["xrange"]).combine_first(
        matched_strings[" range"]).combine_first(
        matched_strings['u"']).combine_first(
        matched_strings['u\'']).combine_first(
        matched_strings["print "]).combine_first(
        matched_strings['unicode(']).combine_first(
        matched_strings['__next__(']).combine_first(
        matched_strings['raw_input('])

    output_df = output_df[['line of code', 'true class','true explanation']]

    # Save the DataFrame to a CSV file
    output_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    input1_path = '/mention/your/local/file_path/python3_1.txt'
    input2_path = '/mention/your/local/file_path/python2_1.txt'
    input3_path = '/mention/your/local/file_path/common_1.txt'
    output_csv_path = '/mention/your/local/file_path/csv_with_trueexp.csv'
    generate_output_csv(input1_path, input2_path, input3_path,output_csv_path)
