import tilus
from tilus.target import nvgpu_sm90
from tilus.testing.requires import requires

class DemoBlockCluster(tilus.Script):
    def __call__(self):
        self.attrs.blocks = [4, 8, 2]
        self.attrs.blocks_per_cluster = [2, 2, 1]
        self.attrs.warps = 4

        self.printf("blockIdx: [%d, %d, %d]\n", self.blockIdx.x, self.blockIdx.y, self.blockIdx.z)

@requires(nvgpu_sm90)
def test_script_blocks_per_cluster():
    kernel = DemoBlockCluster()
    kernel()
