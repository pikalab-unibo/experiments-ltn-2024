import os
import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class NotebookTest(unittest.TestCase):
    
    def test_notebooks(self):
        examples_folder = os.path.join(os.getcwd(), 'examples')
        
        for filename in os.listdir(examples_folder):
            if filename.endswith('.ipynb'):
                notebook_path = os.path.join(examples_folder, filename)
                with open(notebook_path) as f:
                    nb = nbformat.read(f, as_version=4)
                
                ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
                try:
                    ep.preprocess(nb, {'metadata': {'path': examples_folder}})
                except Exception as e:
                    self.fail(f"Error executing notebook: {notebook_path}\n{str(e)}")

if __name__ == '__main__':
    unittest.main()