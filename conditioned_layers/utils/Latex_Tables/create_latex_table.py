from python2latex import Document, Table, build
from utils.Latex_Tables.LatexTable import LatexTable
from utils.Latex_Tables.datasources import ExperimentResultDatasource, ResourceUsageDatasource, CompositeDataSource, DataLoader
import numpy as np
import os

def create_table(output_file, caption, dataloader, table, experiments, columns):
    data = dataloader.load_data(experiments, columns)
    output_string = table(data, caption)

    with open(output_file, mode="w", encoding="utf-8") as f:
        f.write(output_string)


if __name__ == "__main__":
    current_location = os.path.dirname(os.path.realpath(__file__))
    base_path = current_location + "/../../"
    result_path = base_path + "experiments/baseline_hypernet_cifar/results/"
    resources_file = base_path + "utils/resources_estimation/results/baseline"

    data_fields = ["memory", "computation", "train accuracy", "test accuracy"]
    table_columns = [{"name":"Paramètres", "format": ".2e", "highlight": ("low", "bold") },
                     {"name": "Calculs(FLOPS)", "format": ".2e", "highlight": ("low", "bold")},
                     {"name": [["Exactitude"], ["entraînement"]], "format": ".2f", "highlight": ("high", "bold")},
                     {"name": [["Exactitude"], ["test"]], "format": ".4f", "highlight": ("high", "bold")}
                    ]

    baseline_result_path = base_path + "experiments/baseline_hypernet_cifar/results/"
    baseline_resources_file = base_path + "utils/resources_estimation/results/baseline"
    baseline_datasource = CompositeDataSource([ExperimentResultDatasource(baseline_result_path),
                                               ResourceUsageDatasource(baseline_resources_file)])
    baseline_experiments = ["full", "reduced", "fraction_75", "fraction_50", "fraction_25"]
    baseline_rows = ["Modèle complet", "Modèle réduit", "Fraction 25\%", "Fraction 50\%", "Fraction 75\%"]
    baseline_columns = [{"name": "Configuration"}]
    baseline_columns.extend(table_columns)

    phase1_result_path = base_path + "experiments/Policy_phase_one/results/"
    phase1_resources_file = base_path + "utils/resources_estimation/results/phase_one"
    phase1_datasource = CompositeDataSource([ExperimentResultDatasource(phase1_result_path),
                                               ResourceUsageDatasource(phase1_resources_file)])
    phase_one_experiments = ["embeddings_1_factor", "embeddings_2_factor", "embeddings_4_factor"]
    phase_one_rows = ["1", "2", "4"]
    phase_one_columns = [{"name":"Nombre de facteurs"}]
    phase_one_columns.extend(table_columns)

    phase2_result_path = base_path + "experiments/Policy_phase_two/results/"
    phase2_resources_file = base_path + "utils/resources_estimation/results/phase_two"
    phase2_datasource = CompositeDataSource([ExperimentResultDatasource(phase2_result_path),
                                             ResourceUsageDatasource(phase2_resources_file)])
    phase_two_experiments = ["type_cell_gru", "type_cell_lstm"]
    phase_two_rows = ["GRU", "LSTM"]
    phase_two_columns = [{"name": "Type de cellule RNN"}]
    phase_two_columns.extend(table_columns)

    phase3_result_path = base_path + "experiments/Policy_phase_three/results/"
    phase3_resources_file = base_path + "utils/resources_estimation/results/phase_three"
    phase3_datasource = CompositeDataSource([ExperimentResultDatasource(phase3_result_path),
                                             ResourceUsageDatasource(phase3_resources_file)])
    phase_three_experiments = ["layer_count_1", "layer_count_2", "layer_count_4"]
    phase_three_rows = ["1", "2", "4"]
    phase_three_columns = [{"name": "Nombre de cellules RNN"}]
    phase_three_columns.extend(table_columns)

    phase4_result_path = base_path + "experiments/Policy_phase_four/results/"
    phase4_resources_file = base_path + "utils/resources_estimation/results/phase_four"
    phase4_datasource = CompositeDataSource([ExperimentResultDatasource(phase4_result_path),
                                             ResourceUsageDatasource(phase4_resources_file)])
    phase_four_experiments = ["full", "reduced", "fraction_75", "fraction_50", "fraction_25"]
    phase_four_rows = ["Modèle complet", "Modèle réduit", "Fraction 25\%", "Fraction 50\%", "Fraction 75\%"]
    phase_four_columns = [{"name": "Configuration"}]
    phase_four_columns.extend(table_columns)

    create_table("results/baseline_table.tex",
                 "Mesure de la réduction du modèle et du jeu d'entraînement sur les performances de la base de référence",
                 DataLoader(baseline_datasource),
                 LatexTable(baseline_columns, baseline_rows),
                 baseline_experiments,
                 data_fields)

    create_table("results/phase1_table.tex",
                 "Phase 1: Performances et ressources pour différentes valeurs de factorisation des vecteurs de couche",
                 DataLoader(phase1_datasource),
                 LatexTable(phase_one_columns, phase_one_rows),
                 phase_one_experiments,
                 data_fields)

    create_table("results/phase2_table.tex",
                 "Phase 2: Performances et ressources pour différents types de cellule RNN",
                 DataLoader(phase2_datasource),
                 LatexTable(phase_two_columns, phase_two_rows),
                 phase_two_experiments,
                 data_fields)

    create_table("results/phase3_table.tex",
                 "Phase 3: Performances et ressources pour différents nombres de cellules RNN",
                 DataLoader(phase3_datasource),
                 LatexTable(phase_three_columns, phase_three_rows),
                 phase_three_experiments,
                 data_fields)
    #
    # create_table("phase4_table.tex",
    #              "Mesure de la réduction du modèle et du jeu d'entraînement sur les performances du PBCNN",
    #              DataLoader(phase4_datasource),
    #              LatexTable(phase_four_columns, phase_four_rows),
    #              phase_four_experiments,
    #              data_fields)

    # fields_of_interests = ["memory", "computation", "train accuracy", "test accuracy"]
    #
    #
    # headers = ["Configuration", "Paramètres", "Calculs(FLOPS)", [["Exactitude"], ["entraînement"]], [["Exactitude"], ["test"]]]
    # # headers = ["Configuration", "Paramètres", "Calculs(FLOPS)", "Exactitude entraînement", "Exactitude test"]
    # row_names = ["Modèle complet", "Modèle réduit", "Fraction 25\%", "Fraction 50\%", "Fraction 75\%"]
    #
    # dataLoader = DataLoader(CompositeDataSource([ExperimentResultDatasource(result_path), ResourceUsageDatasource(resources_file)]))
    #
    # tabular_data = dataLoader.load_data(baseline_experiments, fields_of_interests)
    #
    # row, col = tabular_data.shape
    #
    # table = Table(shape=(row+1, col+1), alignment='c')
    #
    # table.caption = 'Résultats pour les configurations de la base de référence'
    #
    # table[1:, 1:] = tabular_data
    # for i in range(len(field_formats)):
    #     table[1:, 1 + i].change_format(field_formats[i])
    #
    # table[1:, 0] = row_names
    # for i in range(len(headers)):
    #     header = headers[i]
    #     if isinstance(header, list):
    #         line_count = len(header)
    #         table[0, i].divide_cell(shape=(2, 1), alignment='c')[:] = header
    #     else:
    #         table[0, i] = header
    #
    # table[0, 0:].add_rule(trim_left=True, trim_right='.3em')
    #
    # for column in range(1, 3):
    #     table[1:, column].highlight_best('low', 'bold')  # Best per row, for the last 3 columns
    # for column in range(3, 5):
    #     table[1:, column].highlight_best('high', 'bold')  # Best per row, for the last 3 columns
    #
    # tex = table.build()
    #
    # with open (current_location +"/table.tex", mode="w", encoding="utf-8") as f:
    #     f.write(tex)