This is a script to scrape python programming examples and adding label and  true explaination to it 


Step 1:
first run the getfiles_from_github_repo.py script to generate python3 file
and run 3to2 to convert the python3 script to python2 format and generate a python2 file


step 2:
Run the remove_common_lines.py to remove the common lines in python2 and python3 (be sure to change the file_path according to your file name)


step 3:
run trueexplaination_and_csv_file.py to add true explaination and convert the lines to csv file


step 4:
after getting the text files from the above steps,there are 3 ways you can progress with training and testing the data.
1) label common lines as python3 itself:
   For this use trainandtest.py and modify the file paths as required.
2) label common lines as a different class:
   For this use trainandtestcommonlines2.py and modify the file paths as needed.
3) complety ignore the common lines:
   For this just comment the part which labels the common lines as 1 in trainandtest.py
For all these the prediction will be generated and plots will be stored in the specified file path.
