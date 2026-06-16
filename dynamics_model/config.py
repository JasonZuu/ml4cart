import os

from common.paths import dynamics_generated_dir, dynamics_root

# ======= GENERATE FOLDERS =======

# Raw and generated dynamics data live outside the code package by default.
DATA_DIR = str(dynamics_root())
DATA_DIRS = [DATA_DIR]
GENERATED_DIR = str(dynamics_generated_dir())
MODEL_DIR = os.path.join(GENERATED_DIR, "models")
RESULTS_DIR = os.environ.get("ML4CART_DYNAMICS_RESULTS", "dynamics_model/results")


# ======= CELL TRACKING SETTINGS =======
FIJI_PATH = r"dynamics_data/Fiji" # the folder of ImageJ
JAVA_ARGUMENTS = '-Xmx16g'

SPECIAL_THRESHOLDING = {"DiD-MSLN_NCI6_5 percent 20ms001/XY4" : 130}
CELL_TRACKING_DATASET_CONFIGS = {
    # "CART1": {"images_folder" : f"{DATA_DIR}/20250429_CART_8 patients_day 6_for AI",
    #               "case_name": "CART1",
    #                  "prefix": "CART1_",
    #          "subcase_names" : ["NYU318",
    #                             "NYU352",
    #                             "NYU358",
    #                             "NYU360",
    #                             "NCI2",
    #                             "NCI6",
    #                             "NCI8",
    #                             "NCI9"],
    #     "specific_thresholds": {"DiD-MSLN_NCI6_5 percent 20ms001/XY4" : 130}},

    # "CART2": {"images_folder" : f"{DATA_DIR}/20250522_Meso IL18 CART_5 patients_day 6_for AI",
    #              "case_name": "CART2",
    #                 "prefix": "CART2_",
    #         "subcase_names" : ["NYU352",
    #                            "NYU360",
    #                            "NCI6",
    #                            "NCI8",
    #                            "NCI9"]},

    # "PDO": {"images_folder" : f"{DATA_DIR}/20250710_PDO device 1 to 8_for AI",
    #              "case_name": "PDO",
    #                 "prefix": "PDO_",
    #         "subcase_names" : ["Device1",
    #                            "Device2",
    #                            "Device3",
    #                            "Device4",
    #                            "Device5",
    #                            "Device6",
    #                            "Device7",
    #                            "Device8"]},
    # "Stroma1": {"images_folder" : f"{DATA_DIR}/20251022_NCI 9 Stroma CART dynamics_Round 1_progressive",
    #             "case_name": "Stroma1",
    #             "prefix": "Stroma1_",
    #             "subcase_names" : ["NCI9_Stroma_7",
    #                               "NCI9_Stroma_8",],
    #             "specific_thresholds": {"DiD-CAR T_NCI9_Stroma_7/XY1" : 110,
    #                                     "DiD-CAR T_NCI9_Stroma_8/XY5" : 110,}},
    # "Stroma2": {"images_folder" : f"{DATA_DIR}/20251029_NCI 9 Stroma CART dynamics_Round 2_progressive",
    #             "case_name": "Stroma2",
    #             "prefix": "Stroma2_",
    #             "subcase_names" : ["NCI9_Stroma_7",
    #                               "NCI9_Stroma_8",]},
    "CART3": {"images_folder" : f"{DATA_DIR}/20260114_8 patients_early CAR T",
              "case_name": "CART3",
              "prefix": "CART3_",
              "subcase_names" : ["NYU318",
                                 "NYU352",
                                 "NYU358",
                                 "NYU360",
                                 "NCI2",
                                 "NCI6",
                                 "NCI8",
                                 "NCI9"]},
    "External_r1": {"images_folder" : f"{DATA_DIR}/20251118_3 matched patients_CART and drugs_early dynamics",
                    "case_name": "External_round1",
                    "prefix": "External_round1_",
                    "subcase_names" : ['1_NYU285_IgG ctrl_20ms_2percent','3_NYU285_iAREG 2ug_20ms_2percent',
                                       '1_NYU318_IgG ctrl_20ms_2percent','3_NYU318_iAREG 2ug_20ms_2percent',
                                       '1_NYU774_IgG ctrl_20ms_2percent','3_NYU774_iAREG 2ug_20ms_2percent',
                                       '2_NYU285_AREG Cytokine_20ms_2percent001', '4_NYU285_iAREG 10ug_20ms_2percent',
                                       '2_NYU318_AREG cytokine_20ms_2percent', '4_NYU318_iAREG 10ug_20ms_2percent',
                                       '2_NYU774_AREG Cytokine_20ms_2percent', '4_NYU774_iAREG 10ug_20ms_2percent',],
                "specific_thresholds": {"1_NYU285_IgG ctrl_20ms_2percent/XY2" : 110,
                                        "2_NYU285_AREG Cytokine_20ms_2percent001/XY7": 110},
                    },    
        "External_r2": {"images_folder" : f"{DATA_DIR}/20251211_CAR T and drug_early stage_round 4",
                    "case_name": "External_round2",
                    "prefix": "External_round2_",
                    "subcase_names" : ['DiD-CAR T_CAR T_AREG_50ms_2percent001', 'DiD-CAR T_CAR T_IgG_50ms_2percent',
                                       'DiD-CAR T_CAR T_AREG_iEGFR_50ms_2percent', 'DiD-CAR T_CAR T_iAREG_50ms_2percent',]},
    "Drug": {"images_folder" : f"{DATA_DIR}/20260311_3 matched patients_Round 2",
             "case_name": "Drug",
             "prefix": "Drug_",
             # Group IDs map to: 1.x=NYU285, 2.x=NYU318, 3.x=NYU774
             # x=1: CAR T and IgG (ctrl), x=2: CAR T and iAREG, x=3: FAP CAR T
             "subcase_names": ["1.1", "1.2", "1.3",
                               "2.1", "2.2", "2.3",
                               "3.1", "3.2", "3.3"]},
}





# ======= DATASET GENERATION SETTINGS =======
SEQ_LEN = 100 # Number of frames to use.
DATASET_CONFIGS = {
    "CART1": {"annotation_path" : f"{DATA_DIR}/CART1 annotations.xlsx",
              "data_folder"     : f"{DATA_DIR}/CART1"},
    "CART2": {"annotation_path" : f"{DATA_DIR}/CART2 annotations.xlsx",
              "data_folder"     : f"{DATA_DIR}/CART2"},
    "CART3": {"annotation_path" : f"{DATA_DIR}/CART3 annotations.xlsx",
              "data_folder"     : f"{DATA_DIR}/CART3"},
    "PDO":   {"annotation_path" : f"{DATA_DIR}/PDO annotations.xlsx",
              "data_folder"     : f"{DATA_DIR}/PDO"},
    "Stroma1": {"annotation_path" : f"{DATA_DIR}/Stroma1 annotations.xlsx",
                "data_folder"     : f"{DATA_DIR}/Stroma1"},
    "Stroma2": {"annotation_path" : f"{DATA_DIR}/Stroma2 annotations.xlsx",
                "data_folder"     : f"{DATA_DIR}/Stroma2"},
    "Drug":   {"annotation_path" : f"{DATA_DIR}/20260311_3 matched patients_Round 2/20260308_3 matched patients_PDO size for testing.xlsx",
               "data_folder"     : f"{DATA_DIR}/Drug"},
}

SEQ_DATASET_PREFIX = ""
TRACK_DATASET_PREFIX = ""

features = [ # Time-based Features 
    'AREA', 'PERIMETER', 'CIRCULARITY',
    'ELLIPSE_ASPECTRATIO','SOLIDITY', 
    'SPEED', "MEAN_SQUARE_DISPLACEMENT", #"RADIUS"
]


track_features = [ # Track-Level Statistics Features
    "TRACK_DISPLACEMENT", "TRACK_STD_SPEED",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]

FEATURE_LEN = len(features)
TRACK_LEN = len(track_features)


# ======= TRAINING SETTINGS =======
TEST_TRAIN_SPLIT_ANNOTATION_PATH = r"dynamics_data/data_split.json"
SEQ_DATASET_PATH = os.path.join(GENERATED_DIR, f"{SEQ_DATASET_PREFIX}trajectory_dataset_{SEQ_LEN}.npz")
TRACK_DATASET_PATH = os.path.join(GENERATED_DIR, f"{TRACK_DATASET_PREFIX}track_dataset.npz")

DROPOUT = 0.3
MAX_EPOCHS = 200
BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 20

MIN_POW_FUSION = 4
MAX_POW_FUSION = 12

MIN_POW_HIDDEN = 2
MAX_POW_HIDDEN = 7

ABLATION_CONFIGS = {
    "Specify" : {
        "features": features,
        "track_features" :track_features
    },
}
