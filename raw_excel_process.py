import xlrd
import os
import numpy as np

FIELD_SLICE_DICT = {"装运": "PORT",
             "卸货": "PORT",
             "船名": "TXT",
             "合同": "NO",
             "LC": "LCNO",
             "BL": "BLNO",
             "发票": "INNO"}

class ModelData():
    def __init__(self, field_type="",
                 slice_type="",
                 ocr_info="",
                 type_match_flag=0,
                 loc_info=[],
                 candidate_flag=0):
        self.field_type = field_type
        self.slice_type = slice_type
        self.ocr_info = ocr_info
        self.type_match_flag = type_match_flag
        self.loc_info = loc_info
        self.candidate_flag = candidate_flag

def get_valid_info_from_excel(excel_path):
    raw_excel = xlrd.open_workbook(excel_path, encoding_override='utf-8')
    label_sheet = raw_excel.sheet_by_index(0)
    model_data_list = []
    for i in range(1, label_sheet.nrows):
        region_idx = str(label_sheet.cell_value(i, 0))
        field_type = str(label_sheet.cell_value(i, 1))
        slice_type = str(label_sheet.cell_value(i, 2))
        loc_info = str(label_sheet.cell_value(i, 4))
        ocr_info = str(label_sheet.cell_value(i, 5))

        if region_idx and field_type and slice_type and loc_info and ocr_info:
            locs = [float(loc) for loc in loc_info[1:-1].replace(" ", "").split(',')]
            field_type_start = field_type[:2]
            field_slice_type = FIELD_SLICE_DICT[field_type_start]
            if "其他" in field_type:
                candidate_flag = 0
            else:
                candidate_flag = 1

            if field_slice_type == slice_type:
                type_match_flag = 1
                md = ModelData(field_type, slice_type, ocr_info, type_match_flag, locs, candidate_flag)
                model_data_list.append(md)
            else:
                type_match_flag = 0
                md_pos = ModelData(field_type, slice_type, ocr_info, 1, locs, candidate_flag)
                model_data_list.append(md_pos)
                md_neg = ModelData(field_type, field_slice_type, ocr_info, type_match_flag, locs, candidate_flag)
                model_data_list.append(md_neg)
    # for md in model_data_list:
    #     print("=========================")
    #     print(md.field_type)
    #     print(md.slice_type)
    #     print(md.ocr_info)
    #     print(md.type_match_flag)
    #     print(md.loc_info)
    #     print(md.candidate_flag)
    return model_data_list

def get_all_excel_data(excel_dir):
    trans_id = os.listdir(excel_dir)
    all_data = []
    for id in trans_id:
        trans_dir = os.path.join(excel_dir, id)

        if os.path.isdir(trans_dir):
            res_path = os.path.join(trans_dir, 'result.xlsx')
            model_data_list = get_valid_info_from_excel(res_path)
            all_data.extend(model_data_list)
    return all_data

def get_train_input_label(model_data):
    char_inputs = []
    loc_inputs = []
    char_labels = []
    res_labels = []
    for md in model_data:
        _char = "$" + md.slice_type + "\t" + md.ocr_info + "\n"
        char_inputs.append(_char)
        loc_inputs.append(md.loc_info)
        char_labels.append(md.type_match_flag)
        res_labels.append(md.candidate_flag)
    return np.array(char_inputs), np.array(loc_inputs), np.array(char_labels), np.array(res_labels)


if __name__ == "__main__":
    # excel_path = '/Users/peng_ji/Desktop/labeled01/734201AB19002866/result.xlsx'
    # get_valid_info_from_excel(excel_path)
    excel_dir = '/Users/peng_ji/Desktop/labeled01'
    all_data = get_all_excel_data(excel_dir)
    char_inputs, loc_inputs, char_labels, res_labels = get_train_input_label(all_data)
    #len(char_inputs)
    for i in range(0, 100):
        print(char_inputs[i], char_labels[i])

    # print(len(all_data))
    # print(char_inputs[:5])
    # print(loc_inputs[:5])
    # print(char_labels[:5])
    # print(res_labels[:5])
