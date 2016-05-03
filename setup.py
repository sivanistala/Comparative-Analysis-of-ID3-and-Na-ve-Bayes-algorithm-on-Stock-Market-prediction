application_title = "Comparision of ID3 & Naive Bayes algorithms on stock market data"
main_python_file = "main_page.py"

from distutils.core import setup
import sys

from cx_Freeze import setup.py, main_page.py,recheck,bayesianNetworks,DecisionTree,decisionTrees,id3_naive.py,Node , ProgramGenerator

base = None
if sys.platform == "win32":
    base="Win32GUI"


setup(
        name='FinalYear-Project',
        version='1.0',
        packages=[''],
        url='',
        license='Free use',
        author='Siva',
        author_email='sivanandhalahari@gmail.com',
        description=''
)
