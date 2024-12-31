TASK_INFO = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"
    ]

TASK_INFO_I2V = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality"
    ]

DIM_WEIGHT = {
    "subject consistency":1,
    "background consistency":1,
    "temporal flickering":1,
    "motion smoothness":1,
    "aesthetic quality":1,
    "imaging quality":1,
    "dynamic degree":0.5,
    "object class":1,
    "multiple objects":1,
    "human action":1,
    "color":1,
    "spatial relationship":1,
    "scene":1,
    "appearance style":1,
    "temporal style":1,
    "overall consistency":1
}

DIM_WEIGHT_I2V = {
    "Video-Text Camera Motion": 0.1,
    "Video-Image Subject Consistency": 1,
    "Video-Image Background Consistency": 1,
    "Subject Consistency": 1,
    "Background Consistency": 1,
    "Motion Smoothness": 1,
    "Dynamic Degree": 0.5,
    "Aesthetic Quality": 1,
    "Imaging Quality": 1,
}


NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364}
}

NORMALIZE_DIC_I2V = {
    "Video-Text Camera Motion" :{"Min": 0.0, "Max":1.0 },
    "Video-Image Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Video-Image Background Consistency":{"Min": 0.2615, "Max":1.0 },
    "Subject Consistency":{"Min": 0.1462, "Max": 1.0},
    "Background Consistency":{"Min": 0.2615, "Max": 1.0 },
    "Motion Smoothness":{"Min": 0.7060, "Max": 0.9975},
    "Dynamic Degree":{"Min": 0.0, "Max": 1.0},
    "Aesthetic Quality":{"Min": 0.0, "Max": 1.0},
    "Imaging Quality":{"Min": 0.0, "Max": 1.0},
}

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4
I2V_WEIGHT = 1.0
I2V_QUALITY_WEIGHT = 1.0

QUALITY_LIST = [ 
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"
]

I2V_LIST = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
]

I2V_QUALITY_LIST = [
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
]

I2VKEY={
    "camera_motion":"Video-Text Camera Motion",
    "i2v_subject":"Video-Image Subject Consistency",
    "i2v_background":"Video-Image Background Consistency",
    "subject_consistency":"Subject Consistency",
    "background_consistency":"Background Consistency",
    "motion_smoothness":"Motion Smoothness",
    "dynamic_degree":"Dynamic Degree",
    "aesthetic_quality":"Aesthetic Quality",
    "imaging_quality":"Imaging Quality",
    }