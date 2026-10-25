"""MistNet (Lin et al. 2019): deep-learning separation of rain and biology on NEXRAD.

Not implemented yet: it needs the authors' trained network weights and their 5-elevation
input rendering. This stub documents the interface so the pipeline can list it.
"""

from radarwow.core import Algorithm, Band, Reference, register


@register
class MistNet(Algorithm):
    name = "lin2019_mistnet"
    stage = "qc"
    reference = Reference("T.-Y. Lin, K. Winner, G. Bernstein, A. Mittal, A. M. Dokter, K. G. Horton, C. Nilsson, B. M. Van Doren, A. Farnsworth, F. A. La Sorte, S. Maji, D. Sheldon", 2019,
                          "MistNet: Measuring historical bird migration in the US using archived weather radar data and convolutional neural networks",
                          "Methods Ecol. Evol. 10, 1908-1922", "10.1111/2041-210X.13280")
    bands = frozenset({Band.S})
    requires = ("DBZH", "ZDR", "RHOHV", "VRADH", "WRADH")
    provides = ("BIO_FRACTION",)
    notes = "Stub: port of the pretrained model pending."

    def run_sweep(self, ds, cfg):
        raise NotImplementedError("MistNet needs the pretrained weights; see the reading note (paper 6).")
