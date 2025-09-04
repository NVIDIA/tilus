import tilus
from tilus.target import nvgpu_sm80, nvgpu_sm90
from tilus.testing.requires import requires


class DemoBlockCluster(tilus.Script):
    def __init__(self, cluster_blocks):
        super().__init__()
        self.cluster_blocks = cluster_blocks

    def __call__(self):
        self.attrs.blocks = [2, 2, 2]
        self.attrs.cluster_blocks = self.cluster_blocks
        self.attrs.warps = 4

        self.printf("blockIdx: [%d, %d, %d]\n", self.blockIdx.x, self.blockIdx.y, self.blockIdx.z)


@requires(nvgpu_sm90)
def test_script_blocks_per_cluster_post_sm90():
    kernel = DemoBlockCluster((2, 2, 1))
    kernel()


@requires(nvgpu_sm80)
def test_script_blocks_per_cluster_pre_sm90():
    kernel = DemoBlockCluster((1, 1, 1))
    kernel()
