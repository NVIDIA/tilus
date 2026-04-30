import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tutorials", "matmul-blackwell", "plots"))
from plot_perf import plot_performance

plot_performance()
