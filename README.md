# command-line-grading
Some handy tools for processing essays in a variety of formats and grading them in the command line

# Instructions
## Preparing inputs
A directory containing essays in docx, pages, pdf, and txt formats.
Each filename should contain a unique identifier that matches it to a student.
A csv file containing names and unique identifiers of students.

Note that the function to read .txt files is specific to their structure in my
application and requires modification to read general text files.

[If any of the inputs are .pages files then the script "convert_pages" must be run
first to convert them to .pdfs. This will only work on OSX.]

## Setting directories in the code
Set path to this directory at top of paper_reader.py to PATH + WEEK.
e.g. the input files are in ../data/week1
And set MY_STUDENTS to the path for the csv of student info.

## Running the script
Now in the command line run the script using the command `python paper_reader.py`
The script will then automatically read in the files and create a pandas
dataframe containing columns for the student information, the essay text,
and the input of a grade.

When the files have been read a command line interface will then print each
essay and request you to assign a grade to the essay. A warning will
appear if there is no essay available for a given student.  A counter will show
the number of essays remaining.

For each student simply read the essay and enter the numerical grade.

If you think the text is corrupted or missing then you can assign a missing value
for the student and continue grading. The function grade_change can then be used to
give these students grades after the problem has been investigated (i.e. manually
looking at their submission files)

When all essays have been assigned a grade the dataframe will be saved as a pickle
object. This can be unpickled and converted to a csv as necessary.

The function grade_change also allows you to modify the grade of a particular student
as necessary. You can load this function in Python to use it.

To view the grade file simply open Python and enter the following code:
`import pickle
import pandas as pd
df = pickle.load(open('weekX_graded.p', 'rb')) #where X is the week number
`


## Requirements
The required python packages can all be installed using pip, i.e.
`pip install PyPDF2`

Packages:
`PyPDF2`
`docx`
`pandas`
`numpy`
`copy`
`string`
`re`
