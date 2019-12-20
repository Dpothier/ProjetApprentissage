import re


class lr_t_state_size_extractor():

    def extract_from(self, name):
        match = re.search("lr_([\s\S]*?)_t_([\s\S]*?)_state_size_([\d]*)", name)

        return {
            "lr": float(match[1]),
            "t": float(match[2]),
            "state_size": float(match[3]),
            "seed_size": float(match[3])
        }

class lr_t_state_size_seed_extractor():
    def extract_from(self, name):
        match = re.search("lr_([\s\S]*?)_t_([\s\S]*?)_state_size_([\d]*)_seed_size_([\d]*)", name)

        return {
            "lr": float(match[1]),
            "t": float(match[2]),
            "state_size": float(match[3]),
            "seed_size": float(match[4])
        }