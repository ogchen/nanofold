import pyarrow as pa
from collections import namedtuple
from collections import OrderedDict

from nanofold.common.residue_definitions import RESIDUE_INDEX_MSA_WITH_MASK

MSAMetadata = namedtuple("MSAMetadata", ["pa_type", "feat_size"])
COMPRESSED_MSA_FIELDS = OrderedDict(
    {
        "msa": MSAMetadata(pa.bool_, len(RESIDUE_INDEX_MSA_WITH_MASK)),
        "has_deletion": MSAMetadata(pa.bool_, 1),
        "deletion_value": MSAMetadata(pa.float32, 1),
    }
)
