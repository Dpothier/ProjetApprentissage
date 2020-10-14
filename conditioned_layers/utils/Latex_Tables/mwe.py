from python2latex import Document, Table, build
from utils.Latex_Tables.create_numpy import create_numpy_array
from utils.Latex_Tables.datasources import ExperimentResultDatasource, ResourceUsageDatasource, CompositeDataSource
import numpy as np
import os

if __name__ == "__main__":
    current_location = os.path.dirname(os.path.realpath(__file__))
    base_path = current_location + "/../../"

    row, col = (6, 4)
    column_names =["Those", "are", "column", "names"]
    row_names = ["First", "row", "has", "no", "name"]
    field_formats = ['.2e', '.2e', '.2f']
    tabular_data = np.random.rand(row-1, col-1)

    table = Table(shape=(row, col), alignment='c')

    table[1:, 1:] = tabular_data
    for i in range(len(field_formats)):
        table[1, 1 + i].change_format(field_formats[i])

    table[1:, 0] = row_names
    table[0, 0:] = column_names
    table[1,-1:].divide_cell(shape=(2,1), alignment='c')[:] = [['Longer'],['Title']]

    tex = table.build()

    with open (current_location +"/table2.tex", mode="w", encoding="utf-8") as f:
        f.write(tex)