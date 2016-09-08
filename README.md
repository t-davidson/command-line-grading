# command-line-grading
Some handy tools for processing essays in a variety of formats and grading them in the command line

# Instructions
Inputs: 
A directory containing essays in docx, pages, pdf, and txt formats.
Each filename should contain a unique identifier that matches it to a student.
A csv file containing names and unique identifiers of students.

Note that the function to read .txt files is specific to their structure in my
application and requires modification to read general text files.

[If any of the inputs are .pages files then the script "convert_pages" must be run
first to convert them to .pdfs. This will only work on OSX.]

Set path to this directory at top of paper_reader.py to PATH + WEEK.
e.g. the input files are in ../data/week1
And set MY_STUDENTS to the path for the csv of student info.


The script will then automatically read in the files and create a pandas
dataframe containing columns for the student information, the essay text,
and the input of a grade.

When the files have been read a command line interface will then print each
essay and request you to assign a grade to the essay. A warning will
appear if there is no essay available for a given student.  A counter will show
the number of essays remaining.


When all essays have been assigned a grade the dataframe will be saved as a pickle 
object. This can be unpickled using pickle.load and converted to a csv as necessary.

The function grade_change also allows you to modify the grade of a particular student
as necessary.



