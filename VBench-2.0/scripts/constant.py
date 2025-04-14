TASK_INFO = [
    "Camera Motion",
    "Complex Landscape",
    "Complex Plot",
    "Composition",
    "Diversity",
    "Dynamic Attribute",
    "Dynamic Spatial Relationship",
    "Human Anatomy",
    "Human Clothes",
    "Human Identity",
    "Human Interaction",
    "Instance Preservation",
    "Material",
    "Mechanics",
    "Motion Order Understanding",
    "Motion Rationality",
    "Multi-View Consistency",
    "Thermotics"
    ]

DIM_WEIGHT = {
    "Camera Motion": 1,
    "Complex Landscape": 1,
    "Complex Plot": 1,
    "Composition": 1,
    "Diversity": 1,
    "Dynamic Attribute": 1,
    "Dynamic Spatial Relationship": 1,
    "Human Anatomy": 1,
    "Human Clothes": 1,
    "Human Identity": 1,
    "Human Interaction": 1,
    "Instance Preservation": 1,
    "Material": 1,
    "Mechanics": 1,
    "Motion Order Understanding": 1,
    "Motion Rationality": 1,
    "Multi-View Consistency": 1,
    "Thermotics": 1
}

NORMALIZE_DIC = {
    "Camera Motion": {"Min": 0.0, "Max": 1.0},
    "Complex Landscape": {"Min": 0.0, "Max": 1.0},
    "Complex Plot": {"Min": 0.0, "Max": 1.0},
    "Composition": {"Min": 0.0, "Max": 1.0},
    "Diversity": {"Min": 0.0, "Max": 1.0},
    "Dynamic Attribute": {"Min": 0.0, "Max": 1.0},
    "Dynamic Spatial Relationship": {"Min": 0.0, "Max": 1.0},
    "Human Anatomy": {"Min": 0.0, "Max": 1.0},
    "Human Clothes": {"Min": 0.0, "Max": 1.0},
    "Human Identity": {"Min": 0.0, "Max": 1.0},
    "Human Interaction": {"Min": 0.0, "Max": 1.0},
    "Instance Preservation": {"Min": 0.0, "Max": 1.0},
    "Material": {"Min": 0.0, "Max": 1.0},
    "Mechanics": {"Min": 0.0, "Max": 1.0},
    "Motion Order Understanding": {"Min": 0.0, "Max": 1.0},
    "Motion Rationality": {"Min": 0.0, "Max": 1.0},
    "Multi-View Consistency": {"Min": 0.0, "Max": 1.0},
    "Thermotics": {"Min": 0.0, "Max": 1.0}
}

CREATIVITY_LIST = [
    "Composition",
    "Diversity"
    ]

COMMONSENSE_LIST = [
    "Instance Preservation",
    "Motion Rationality"
    ]

CONTROLLABILITY_LIST = [
    "Camera Motion",
    "Complex Landscape",
    "Complex Plot",
    "Dynamic Attribute",
    "Dynamic Spatial Relationship",
    "Human Interaction",
    "Motion Order Understanding"
    ]

HUMAN_FIDELITY_LIST = [
    "Human Anatomy",
    "Human Clothes",
    "Human Identity"
    ]

PHYSICS_LIST = [
    "Material",
    "Mechanics",
    "Multi-View Consistency",
    "Thermotics"
    ]