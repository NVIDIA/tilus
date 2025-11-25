import pandas
import tilus
import torch
from tilus import float16, float32, int32, uint32
from tilus.ir.tensor import GlobalTensor, RegisterTensor
from tilus.utils import benchmark_func, cdiv


class Pipeline(tilus.Class):
    def __init__(
        self, num_stages: int, producer_arrive_count: int, consumer_arrive_count: int
    ):
        self.num_stages: int = num_stages
        self.full_barriers = self.mbarrier.alloc(
            [consumer_arrive_count for _ in range(num_stages)]
        )
        self.empty_barriers = self.mbarrier.alloc(
            [producer_arrive_count for _ in range(num_stages)]
        )
        self.producer_stage: int32 = 0
        self.consumer_stage: int32 = 0
        self.producer_phase: uint32 = self.mbarrier.producer_initial_phase
        self.consumer_phase: uint32 = self.mbarrier.consumer_initial_phase

    def producer_acquire(self):
        self.mbarrier.wait(
            barrier=self.full_barriers[self.producer_stage], phase=self.producer_phase
        )

    def producer_advance(self):
        self.producer_stage = (self.producer_stage + 1) % self.num_stages
        self.producer_phase = self.producer_phase ^ (self.producer_stage == 0)

    def producer_release_barrier(self) -> RegisterTensor:
        return self.empty_barriers[self.producer_stage]

    def consumer_acquire(self):
        self.mbarrier.wait(
            barrier=self.empty_barriers[self.consumer_stage], phase=self.consumer_phase
        )

    def consumer_advance(self):
        self.consumer_stage = (self.consumer_stage + 1) % self.num_stages
        self.consumer_phase = self.consumer_phase ^ (self.consumer_stage == 0)

    def consumer_release_barrier(self) -> RegisterTensor:
        return self.full_barriers[self.consumer_stage]
