"""Constants for the target discovery workflow."""

ANCHOR_GENES: tuple[str, ...] = ()
IFNG_FOCUS_GENES: tuple[str, ...] = ()

CELLTYPES = (
    "Fibroblast_S1", "Fibroblast_S2", "Fibroblast_S3",
    "Macrophage", "Macrophage_cycling",
    "Pericyte",
    "T_cell_CD4", "T_cell_CD8", "T_cell_CD8_cycling", "T_cell_regulatory",
    "NK",
    "cDC1", "cDC2", "DC_mature", "pDC",
    "Neutrophil", "Mast_cell",
    "Monocyte_classical",
    "Endothelial_venous", "Endothelial_arterial",
)

TYPE_MAPPING = {
    "Fibroblast_S1": "CAF", "Fibroblast_S2": "CAF", "Fibroblast_S3": "CAF",
    "Macrophage": "TAM", "Macrophage_cycling": "TAM",
    "Pericyte": "Stromal",
    "T_cell_CD4": "CD4T", "T_cell_CD8": "CD8T",
    "T_cell_CD8_cycling": "CD8T", "T_cell_regulatory": "Treg",
    "NK": "NK",
    "cDC1": "DC", "cDC2": "DC", "DC_mature": "DC", "pDC": "DC",
    "Neutrophil": "Neutrophil", "Mast_cell": "Mast",
    "Monocyte_classical": "Monocyte",
    "Endothelial_venous": "Endothelial", "Endothelial_arterial": "Endothelial",
}

ST_DECONV_MAP = {
    "Fibroblast_S1": ("Fibro_ADAMDEC1", "Fibro_CXCL8", "Fibro_CXCL14"),
    "Fibroblast_S2": ("Fibro_GPM6B", "Fibro_KCNN3", "Fibro_MYH11"),
    "Fibroblast_S3": ("Fibro_NOTCH3", "Fibro_PI16"),
    "Macrophage": ("Mac_M1", "Mac_M2", "Mac_SPP1"),
    "Macrophage_cycling": ("Mac_M1",),
    "Pericyte": ("Endo",),
    "T_cell_CD4": ("CD4_CXCL13", "CD4_Tcm", "CD4_Treg", "CD4_act"),
    "T_cell_CD8": ("CD8_Cyto", "CD8_HSP", "CD8_Teff", "CD8_Tem", "CD8_Tex"),
    "T_cell_CD8_cycling": ("CD8_Cyto",),
    "T_cell_regulatory": ("CD4_Treg",),
    "NK": ("NK_gdT",),
    "cDC1": ("cDC1",), "cDC2": ("cDC2",), "DC_mature": ("DC_LAMP3",), "pDC": ("pDC",),
    "Neutrophil": ("Monocyte_S100A8",),
    "Mast_cell": ("Mast",),
    "Monocyte_classical": ("Monocyte_S100A8",),
    "Endothelial_venous": ("Endo",), "Endothelial_arterial": ("Endo",),
}

ICB_TO_NEU_MAP = {
    "Fibro": ("Fibroblast_S1", "Fibroblast_S2", "Fibroblast_S3"),
    "Mph": ("Macrophage", "Macrophage_cycling"),
    "CD8": ("T_cell_CD8", "T_cell_CD8_cycling"),
    "T": ("T_cell_CD4", "T_cell_CD8", "T_cell_regulatory"),
    "Endo": ("Endothelial_venous", "Endothelial_arterial"),
    "Pericyte": ("Pericyte",),
    "Tumor": (),
    "Coloncyte": (), "Goblet": (), "Glia": (), "Tuft": (),
}

PRIOR_AXES = (
    ("CAF", "TAM", 0.3),
    ("CAF", "Treg", 0.3),
    ("TAM", "CD8T", 0.3),
    ("DC", "CD8T", 0.2),
    ("Neutrophil", "TAM", 0.2),
    ("CAF", "Endothelial", 0.2),
)

SCORE_WEIGHTS = {
    "causal": 0.25,
    "spatial": 0.25,
    "consistency": 0.25,
    "actionability": 0.10,
    "niche": 0.15,
}
