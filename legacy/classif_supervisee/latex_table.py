def output_to_table(filename, data, rowname_column_header, columnnames, rownames, caption, label):

    table = \
        "\\begin{table}[] \n" + \
            "\\begin{center}\n" + \
            "\\caption{{{}}}\n".format(caption) + \
            "\\label{{{}}}\n".format(label) + \
            "\\begin{tabular}\n"

    column_format = "{ | r |"
    for i in range(0, data.shape[1]):
        column_format += " r |"
    column_format += "}\n"
    table += column_format
    table += "\\hline\n"

    column_headings = "{} &".format(rowname_column_header)
    for index in range(0, len(columnnames)):
        column_headings += columnnames[index]
        if index != len(columnnames) - 1:
            column_headings += " & "
    column_headings += " \\\\\n"

    table += column_headings
    table += "\\hline\n"

    for row in range(0, len(rownames)):
        table_line = "{} & ".format(rownames[row])
        for column in range(0, len(columnnames)):
            table_line += str(data[row, column])
            if column != len(columnnames) - 1:
                table_line += " & "
        table_line += " \\\\\n"
        table += table_line

    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"

    with open('./results/{}.txt'.format(filename), "w") as f:
        f.write(table)


def output_to_table_3d(filename, data, rowname_columns_headers, columnnames, first_variable_rownames,
                       second_variable_rownames, caption, label):

    table = \
        "\\begin{table}[] \n" + \
            "\\begin{center}\n" + \
            "\\caption{{{}}}\n".format(caption) + \
            "\\label{{{}}}\n".format(label) + \
            "\\begin{tabular}\n"

    column_format = "{ | r | r |"
    for i in range(0, columnnames):
        column_format += " r |"
    column_format += "}\n"
    table += column_format
    table += "\\hline\n"

    column_headings = ""
    for header in rowname_columns_headers:
        column_headings += "{} & ".format(header)
    for index in range(0, len(columnnames)):
        column_headings += columnnames[index]
        if index != len(columnnames) - 1:
            column_headings += " & "
    column_headings += " \\\\\n"

    table += column_headings
    table += "\\hline\n"

    for row in range(0, len(first_variable_rownames)):
        first_variable_header_row = "{} & &".format(first_variable_rownames[row])
        first_variable_header_row = "{} & ".format(first_variable_rownames[row])
        for column in range(0, len(columnnames)):
            first_variable_header_row += str(data[row, column])
            if column != len(columnnames) - 1:
                first_variable_header_row += " & "
        first_variable_header_row += " \\\\\n"
        table += first_variable_header_row

    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"

    with open('./results/{}.txt'.format(filename), "w") as f:
        f.write(table)
