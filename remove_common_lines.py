def remove_common_lines(file1_path, file2_path, output1_path,output2_path,output3_path):
    with open(file1_path, 'r') as file1:
        lines1 = [line.strip() for line in file1.readlines() if line.strip()]

    with open(file2_path, 'r') as file2:
        lines2 = [line.strip() for line in file2.readlines() if line.strip()]

    # Find the common lines
    common_lines = set(lines1) & set(lines2)

    # Keep lines that are not common
    different_lines1 = [line for line in lines1 if line not in common_lines]
    different_lines2 = [line for line in lines2 if line not in common_lines]
    # Sort modified lines alphabetically
    different_lines1.sort()
    different_lines2.sort()
    with open(output1_path, 'w') as output1:
        output1.write('\n'.join(different_lines1))

    with open(output2_path, 'w') as output2:
        output2.write('\n'.join(different_lines2))
    with open(output3_path, 'w') as output3:
        output3.write('\n'.join(common_lines))

    # Return the number of lines in each file after writing
    return len(different_lines1), len(different_lines2), len(common_lines)

if __name__ == "__main__":
    file1_path = '/python3_file_new.txt'
    file2_path = '/python2_file_new.txt'
    output1_path = '/python3.txt'
    output2_path = '/python2.txt'
    output3_path = '/common.txt'

    python3_nof_lines, python2_nof_lines,common_nof_lines = remove_common_lines(file1_path, file2_path, output1_path, output2_path,output3_path)

    print(f"Number of lines in python3 file: {python3_nof_lines}")
    print(f"Number of lines in python2 file: {python2_nof_lines}")
    print(f"Number of lines in common file: {common_nof_lines}")
