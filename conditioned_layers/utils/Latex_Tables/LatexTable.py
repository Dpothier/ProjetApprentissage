from python2latex import Table
import numpy as np

class LatexTable:
    def __init__(self, columns: list, row_names: list):
        self.columns = columns
        self.row_names = row_names

    def __call__(self, data: np.array, caption):
        row, col = data.shape

        table = Table(shape=(row + 1, col + 1), alignment='c')

        table.caption = caption

        table[1:, 1:] = data
        for i in range(len(self.columns)):
            if "format" in self.columns[i]:
                table[1:, i].change_format(self.columns[i]["format"])
            if "highlight" in self.columns[i]:
                table[1:, i].highlight_best(self.columns[i]["highlight"][0], self.columns[i]["highlight"][1])

        table[1:, 0] = self.row_names
        for i in range(len(self.columns)):
            name = self.columns[i]["name"]
            if isinstance(name, list):
                line_count = len(name)
                table[0, i].divide_cell(shape=(line_count, 1), alignment='c')[:] = name
            else:
                table[0, i] = name

        table[0, 0:].add_rule(trim_left=True, trim_right='.3em')

        tex = table.build()

        return tex