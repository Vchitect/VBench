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
    "overall consistency"]

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

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4

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