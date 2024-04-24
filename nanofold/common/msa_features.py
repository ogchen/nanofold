import pyarrow as pa
from collections import namedtuple
from collections import OrderedDict

from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA_WITH_MASK

MSAMetadata = namedtuple("MSAMetadata", ["pa_type", "feat_size"])
MSA_FIELDS = OrderedDict(
    {
        "cluster_msa": MSAMetadata(pa.bool_, len(RESIDUE_INDEX_MSA_WITH_MASK)),
        "cluster_has_deletion": MSAMetadata(pa.bool_, 1),
        "cluster_deletion_value": MSAMetadata(pa.float32, 1),
        "cluster_deletion_mean": MSAMetadata(pa.float32, 1),
        "cluster_profile": MSAMetadata(pa.float32, len(RESIDUE_INDEX_MSA_WITH_MASK)),
        "extra_msa": MSAMetadata(pa.bool_, len(RESIDUE_INDEX_MSA_WITH_MASK)),
        "extra_msa_has_deletion": MSAMetadata(pa.bool_, 1),
        "extra_msa_deletion_value": MSAMetadata(pa.float32, 1),
    }
)
