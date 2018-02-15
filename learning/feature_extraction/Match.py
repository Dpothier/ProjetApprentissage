

def build_matches(bdrv, carcomplaints, matches):
    matched_rows = []
    for match in matches:
        bdrv_row = next(filter(lambda row: row['recall_id'] == match['bdrv_id'], bdrv))
        carcomplaint_row = next(filter(lambda row: row['id'] == match['carcomplaints_id'], carcomplaints))
        matched_rows.append(BDRVCarcomplaintsMatch(bdrv_row, carcomplaint_row, match['is_match']))

    return matched_rows


class BDRVCarcomplaintsMatch:

    def __init__(self, bdrv, carcomplaint, is_match):
        self.bdrv = bdrv
        self.carcomplaint = carcomplaint
        self.is_match = is_match


    def get_bdrv_attribute(self, attribute_name):
        attribute_list = []