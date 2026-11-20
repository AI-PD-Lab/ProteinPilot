import numpy as np

from foundry.constants import TIP_BY_RESTYPE

TIP_BY_RESTYPE

REQUIRED_CONDITIONING_ANNOTATION_VALUES = {
    "is_motif_atom_with_fixed_seq": True,
    "is_motif_atom_with_fixed_coord": True,
    "is_motif_atom_unindexed": False,
    "is_motif_atom_unindexed_motif_breakpoint": False,
}
REQUIRED_CONDITIONING_ANNOTATIONS = list(REQUIRED_CONDITIONING_ANNOTATION_VALUES.keys())
REQUIRED_INFERENCE_ANNOTATIONS = REQUIRED_CONDITIONING_ANNOTATIONS + ["src_component"]
"""Annotations assigned to every valid atom array"""

OPTIONAL_CONDITIONING_VALUES = {
    "is_atom_level_hotspot": 0,
    "is_helix_conditioning": 0,
    "is_sheet_conditioning": 0,
    "is_loop_conditioning": 0,
    "active_donor": 0,
    "active_acceptor": 0,
    "rasa_bin": 3,
    "ref_plddt": 0,
    "is_non_loopy": 0,
    "ss_ratio": -1.0,
    "partial_t": np.nan,
    "ss_label_conditioning": 3,
    "is_motif_token": 1,
    "is_motif_atom": 1,
}
"""Optional conditioning annotations and their default values if not provided."""

CONDITIONING_VALUES = (
    REQUIRED_CONDITIONING_ANNOTATION_VALUES | OPTIONAL_CONDITIONING_VALUES
)
"""Annotations that must be present in the AtomArray at inference time."""

INFERENCE_ANNOTATIONS = REQUIRED_INFERENCE_ANNOTATIONS + list(
    OPTIONAL_CONDITIONING_VALUES.keys()
)
"""All annotations that might be desired at inference time. Determines what AtomArray annotations will be preserved."""

SAVED_CONDITIONING_ANNOTATIONS = [
    "is_motif_atom_with_fixed_seq",
]
"""Annotations for conditioning to save in output files"""

ccd_ordering_atomchar = {
    'TRP': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"),
    'HIS': (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None),
    'TYR': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None),
    'PHE': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None),
    'ASN': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None),
    'ASP': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None),
    'GLN': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None),
    'GLU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None),
    'CYS': (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None),
    'SER': (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None),
    'THR': (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None),
    'LEU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None),
    'VAL': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None),
    'ILE': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None),
    'MET': (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None),
    'LYS': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None),
    'ARG': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None),
    'PRO': (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None),
    'ALA': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
    'GLY': (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None),
    'UNK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
    'MSK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
}
"""Canonical ordering of amino acid atom names in the CCD."""

symmetric_atomchar = {
    "TYR": [[" CE1", " CE2"], [" CD1", " CD2"]],
    "PHE": [[" CE1", " CE2"], [" CD1", " CD2"]],
    "ASP": [[" OD1", " OD2"]],
    "GLU": [[" OE1", " OE2"]],
    "LEU": [[" CD1", " CD2"]],
    "VAL": [[" CG1", " CG2"]],
}
"""Maps residues to their pairs of aton names corresponding to symmetric atoms."""

association_schemes = {
    'atom14': {
        'TRP': (" N  "," CA "," C  "," O  "," CB "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," CG "),
        'HIS': (" N  "," CA "," C  "," O  "," CB "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," CG "),
        'TYR': (" N  "," CA "," C  "," O  "," CB "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," CG "),
        'PHE': (" N  "," CA "," C  "," O  "," CB "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," CG "),

        'ASN': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None," ND2"," OD1",  None,  None," CG "),
        'ASP': (" N  "," CA "," C  "," O  "," CB ",  None,  None," OD1"," OD2",  None,  None,  None,  None," CG "),
        'GLN': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None," NE2"," OE1",  None," CD "," CG "),
        'GLU': (" N  "," CA "," C  "," O  "," CB ",  None,  None," OE2"," OE1",  None,  None,  None," CD "," CG "),

        'CYS': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None," SG ",  None,  None,  None),
        'SER': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None," OG ",  None,  None),
        'THR': (" N  "," CA "," C  "," O  "," CB "," CG2",  None,  None,  None,  None,  None," OG1",  None,  None),

        'LEU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None),
        'VAL': (" N  "," CA "," C  "," O  "," CB "," CG1",  None,  None," CG2",  None,  None,  None,  None,  None),
        'ILE': (" N  "," CA "," C  "," O  "," CB "," CG1"," CD1",  None," CG2",  None,  None,  None,  None,  None),

        'MET': (" N  "," CA "," C  "," O  "," CB "," CG ",  None," CE ",  None,  None," SD ",  None,  None,  None),
        'LYS': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE ",  None," NZ ",  None,  None,  None,  None),

        'ARG': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," NH1"," CZ "," NH2",  None,  None,  None),
        'PRO': (" N  "," CA "," C  "," O  "," CB "," CG ",  None,  None,  None,  None,  None,  None," CD ",  None),

        'UNK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
        'ALA': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
        'MSK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
        'GLY': (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None),
    },

    "permute_ambiguous_only": {

        'CYS': (" N  "," CA "," C  "," O  "," CB ",  None, " SG ", None,  None,  None,  None,  None,  None,  None),
        'ASP': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1", None, " OD2",  None,  None,  None,  None,  None),
        'GLU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1", None, " OE2",  None,  None,  None,  None),

        'SER': (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None),
        'ASN': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None),
        'GLN': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None),

        'TRP': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"),
        'HIS': (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None),
        'TYR': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None),
        'PHE': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None),
        'THR': (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None),
        'LEU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None),
        'VAL': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None),
        'ILE': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None),
        'MET': (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None),
        'LYS': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None),
        'ARG': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None),
        'PRO': (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None),
        'ALA': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
        'GLY': (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None),
        'UNK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
        'MSK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),
    },

    'ccd': ccd_ordering_atomchar,
}
association_schemes['atom14-new'] = association_schemes['atom14'].copy()
association_schemes['atom14-new'] |= {
        'GLN': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None," NE2"," OE1",  None," CG "," CD "),
        'GLU': (" N  "," CA "," C  "," O  "," CB ",  None,  None," OE2"," OE1",  None,  None,  None," CG "," CD "),
        'HIS': (" N  "," CA "," C  "," O  "," CB "," ND1"," CD2"," CE1",  None,  None,  None," NE2",  None," CG "),
}
association_schemes['dense'] = association_schemes['permute_ambiguous_only'].copy()

VIRTUAL_ATOM_ELEMENT_NAME = "VX"
"""The element name annotation that will be assigned to virtual atoms"""

ATOM14_ATOM_NAMES = np.array(
    ["N", "CA", "C", "O", "CB"] + [f"V{i}" for i in range(14 - 5)]
)
"""Atom14 atom names (e.g. CA, V1)"""

ATOM14_ATOM_ELEMENTS = np.array(
    ["N", "C", "C", "O", "C"] + [VIRTUAL_ATOM_ELEMENT_NAME for i in range(14 - 5)]
)
"""Atom14 element names (e.g. C, VX)"""

ATOM14_ATOM_NAME_TO_ELEMENT = {
    name: elem for name, elem in zip(ATOM14_ATOM_NAMES, ATOM14_ATOM_ELEMENTS)
}
"""Mapping from atom14 atom names (e.g. CA, V1) to their corresponding element names (e.g. C, VX)"""

strip_list = lambda x: [(x.strip() if x is not None else None) for x in x]

association_schemes_stripped = {
    name: {k: strip_list(v) for k, v in scheme.items()}
    for name, scheme in association_schemes.items()
}
SELECTION_PROTEIN = ["POLYPEPTIDE(D)", "POLYPEPTIDE(L)"]
SELECTION_NONPROTEIN = [
    "POLYDEOXYRIBONUCLEOTIDE",
    "POLYRIBONUCLEOTIDE",
    "PEPTIDE NUCLEIC ACID",
    "OTHER",
    "NON-POLYMER",
    "CYCLIC-PSEUDO-PEPTIDE",
    "MACROLIDE",
    "POLYDEOXYRIBONUCLEOTIDE/POLYRIBONUCLEOTIDE HYBRID",
]
