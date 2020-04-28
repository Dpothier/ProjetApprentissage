import os
from utils.ressources_estimation.operation.networks import Static_Hypernetwork, Policy_Hypernetwork
from yaml import dump

def produce_baseline_results(output):
    full = Static_Hypernetwork(64, 16, 32, 32, 18, 3, 10)
    reduced = Static_Hypernetwork(16, 8, 32, 32, 18, 3, 10)
    fraction_25 = Static_Hypernetwork(64, 16, 32, 32, 18, 3, 10)
    fraction_50 = Static_Hypernetwork(64, 16, 32, 32, 18, 3, 10)
    fraction_75 = Static_Hypernetwork(64, 16, 32, 32, 18, 3, 10)

    yaml = {
        "full": {
            "memory": full.memory(),
            "computation": full.computation()
        },
        "reduced": {
            "memory": reduced.memory(),
            "computation": reduced.computation()
        },
        "fraction_25": {
            "memory": fraction_25.memory(),
            "computation": fraction_25.computation()
        },
        "fraction_50": {
            "memory": fraction_50.memory(),
            "computation": fraction_50.computation()
        },
        "fraction_75": {
            "memory": fraction_75.memory(),
            "computation": fraction_75.computation()
        }

    }

    with open(output, mode="w", encoding="utf-8") as f:
        dump(yaml, f)


def produce_phase_one_results(output):
    embeddings_1_factor = Policy_Hypernetwork(32, 32, 18, 3, 64, 1, 16, 2, 10, "gru", 1)
    embeddings_2_factor = Policy_Hypernetwork(32, 32, 18, 3, 32, 2, 16, 2, 10, "gru", 1)
    embeddings_4_factor = Policy_Hypernetwork(32, 32, 18, 3, 16, 4, 16, 2, 10, "gru", 1)

    yaml = {
        "embeddings_1_factor": {
            "memory": embeddings_1_factor.memory(),
            "computation": embeddings_1_factor.computation()
        },
        "embeddings_2_factor": {
            "memory": embeddings_2_factor.memory(),
            "computation": embeddings_2_factor.computation()
        },
        "embeddings_4_factor": {
            "memory": embeddings_4_factor.memory(),
            "computation": embeddings_4_factor.computation()
        }
    }

    with open(output, mode="w", encoding="utf-8") as f:
        dump(yaml, f)


def produce_phase_two_results(output):
    gru_cell = Policy_Hypernetwork(32, 32, 18, 3, 16, 4, 16, 2, 10, "gru", 1)
    lstm_cell = Policy_Hypernetwork(32, 32, 18, 3, 16, 4, 16, 2, 10, "lstm", 1)

    yaml = {
        "gru_cell": {
            "memory": gru_cell.memory(),
            "computation": gru_cell.computation()
        },
        "lstm_cell": {
            "memory": lstm_cell.memory(),
            "computation": lstm_cell.computation()
        }
    }

    with open(output, mode="w", encoding="utf-8") as f:
        dump(yaml, f)

def produce_phase_three_results(output):
    state_update_1_layer = Policy_Hypernetwork(32, 32, 18, 3, 16, 4, 16, 2, 10, "gru", 1)
    state_update_2_layer = Policy_Hypernetwork(32, 32, 18, 3, 16, 4, 16, 2, 10, "gru", 2)
    state_update_4_layer = Policy_Hypernetwork(32, 32, 18, 3, 16, 4, 16, 2, 10, "gru", 4)

    yaml = {
        "state_update_1_layer": {
            "memory": state_update_1_layer.memory(),
            "computation": state_update_1_layer.computation()
        },
        "state_update_2_layer": {
            "memory": state_update_2_layer.memory(),
            "computation": state_update_2_layer.computation()
        },
        "state_update_4_layer": {
            "memory": state_update_4_layer.memory(),
            "computation": state_update_4_layer.computation()
        }
    }

    with open(output, mode="w", encoding="utf-8") as f:
        dump(yaml, f)

if __name__ == '__main__':
    output_folder_base = os.path.dirname(os.path.realpath(__file__)) + "/results/"

    produce_baseline_results(output_folder_base + "baseline")
    produce_phase_one_results(output_folder_base + "phase_one")
    produce_phase_two_results(output_folder_base + "phase_two")
    produce_phase_three_results(output_folder_base + "phase_three")