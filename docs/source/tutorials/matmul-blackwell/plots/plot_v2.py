import os
import sys

# plot_basedir is set to docs/source/ in conf.py, and plot_directive
# sets the cwd to plot_basedir before running the script
sys.path.insert(0, os.path.join(os.getcwd(), "tutorials", "matmul-blackwell", "plots"))
from plot_perf import plot_performance

plot_performance(up_to_version=2)
