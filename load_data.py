import pandas as pd
import os


class LoadData:
    def __init__(self, filename, sheetname=None, index_col=None):
        self.file = filename
        self.directory = os.path.dirname(os.path.realpath(__file__))
        self.path = os.path.join(self.directory, self.file)
        if sheetname is None:
            self.df = pd.read_excel(self.path)
        else:
            self.df = pd.read_excel(self.path, sheet_name=sheetname, index_col=index_col)

    def get_data(self):
        return self.df
    