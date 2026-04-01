---
name: ncu-report
description: Analyze NVIDIA Nsight Compute (ncu) profiling reports (.ncu-rep files). Extract metrics, performance data, SASS/CUDA source, and identify bottlenecks.
user_invocable: true
---

# Nsight Compute Report Analysis

You are analyzing an `.ncu-rep` profiling report using the `ncu` CLI (NVIDIA Nsight Compute).
All `ncu` commands MUST use `TMPDIR=/tmp/ncu_tmp` prefix to avoid temp file errors.

The user will provide a path to an `.ncu-rep` file and optionally specify what they want to analyze.
If the user does not specify a report path, check for `.ncu-rep` files under `examples/` in the repo.

## Analysis Workflow

Follow this sequence. Skip steps the user doesn't need, but always start with Step 1.

### Step 1: Overview — List kernels and session info

Run these in parallel:

```bash
# List all kernels with timing
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page raw --csv --metrics gpu__time_duration.sum 2>&1

# Session/device info
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page session --csv 2>&1
```

Present a summary table:
- Kernel name (shortened), Block Size, Grid Size, Duration (ms)
- Device name, compute capability, CUDA version

### Step 2: Speed of Light — Top-level throughput

```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page details --csv --section SpeedOfLight 2>&1
```

Key metrics to highlight per kernel:
- **Duration** (ms)
- **Compute (SM) Throughput** (%) — how busy the SMs are
- **Memory Throughput** (%) — overall memory utilization
- **DRAM Throughput** (%) — HBM bandwidth utilization
- **L1/TEX Cache Throughput** (%)
- **L2 Cache Throughput** (%)
- **SOLBottleneck rule** — check the Rule Description column for bottleneck guidance

### Step 3: Compute & Memory Workload Analysis

```bash
# Compute workload
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page details --csv --section ComputeWorkloadAnalysis 2>&1

# Memory workload
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page details --csv --section MemoryWorkloadAnalysis 2>&1
```

Key compute metrics: Executed IPC Active, SM Busy %, Issue Slots Busy %
Key memory metrics: Mem Busy %, Max Bandwidth %, L1/L2 hit rates

### Step 4: Occupancy

```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page details --csv --section Occupancy 2>&1
```

Report: Theoretical Occupancy, Achieved Occupancy, and limiters (registers, shared memory, block size).

### Step 5: Detailed metrics (on demand)

To extract specific raw metrics:
```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page raw --csv --metrics <metric1>,<metric2>,... 2>&1
```

To filter by kernel:
```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page raw --csv --metrics <metrics> --kernel-name regex:<pattern> 2>&1
```

### Step 6: Source-level analysis (on demand)

SASS-only (default, always available):
```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page source --csv --kernel-name regex:<pattern> 2>&1
```

CUDA source correlated with SASS (requires `--import-source yes` during profiling):
```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page source --csv --print-source cuda,sass --kernel-name regex:<pattern> 2>&1
```

Source output columns include per-instruction: Warp Stall Sampling, Instructions Executed, Thread Instructions Executed, stall reasons (stall_barrier, stall_math, stall_wait, etc.), shared memory conflicts, and more.

### Step 7: Rules / automated analysis (on demand)

Rules are included in the details page output. Look for non-empty "Rule Name" column entries.
```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page details --csv --print-rule-details 2>&1 | grep -v '^"[0-9]' | head -5  # header
```

To see all rule results with descriptions:
```bash
TMPDIR=/tmp/ncu_tmp ncu -i <REPORT> --page details --csv --print-rule-details 2>&1
```
Filter for rows where column 17 (Rule Name) is non-empty.

## Reference: ncu CLI Options for Report Analysis

### Pages (`--page`)
| Page | Description |
|------|-------------|
| `details` | Sections with metrics organized by section name + rule results |
| `raw` | All collected metrics as flat columns (one row per kernel) |
| `source` | Per-instruction source code with correlated metrics |
| `session` | Session info, device attributes, launch settings |

### Key Flags
| Flag | Description |
|------|-------------|
| `--csv` | Output as CSV (essential for parsing) |
| `--metrics <m1>,<m2>` | Filter specific metrics (for `raw` page) |
| `--section <id>` | Filter by section identifier (for `details` page) |
| `--kernel-name regex:<pat>` | Filter kernels by name regex |
| `--kernel-name <exact>` | Filter by exact kernel name |
| `--print-source sass\|ptx\|cuda\|cuda,sass` | Select source view for `source` page |
| `--print-details header\|body\|all` | Control detail level: `header` (default), `body` (charts/tables), `all` |
| `--print-metric-name name` | Show internal metric names instead of display labels |
| `--print-metric-name label-name` | Show both label and internal name |
| `--print-units base` | Show metrics in base units (no auto-scaling) |
| `--print-summary per-kernel` | Aggregate across invocations per kernel (min/max/avg) |
| `--print-rule-details` | Include additional rule tables and KPI metrics |

### Section Identifiers
| Identifier | Display Name |
|------------|-------------|
| `SpeedOfLight` | GPU Speed Of Light Throughput |
| `ComputeWorkloadAnalysis` | Compute Workload Analysis |
| `MemoryWorkloadAnalysis` | Memory Workload Analysis |
| `MemoryWorkloadAnalysis_Tables` | Memory Workload Analysis Tables |
| `Occupancy` | Occupancy |
| `LaunchStats` | Launch Statistics |
| `SchedulerStats` | Scheduler Statistics |
| `WarpStateStats` | Warp State Statistics |
| `InstructionStats` | Instruction Statistics |
| `SourceCounters` | Source Counters |
| `WorkloadDistribution` | GPU and Memory Workload Distribution |
| `NumaAffinity` | NUMA Affinity |
| `SpeedOfLight_RooflineChart` | GPU Speed Of Light Roofline Chart |
| `SpeedOfLight_HierarchicalTensorRooflineChart` | Roofline Chart (Tensor Core) |
| `SpeedOfLight_HierarchicalHalfRooflineChart` | Roofline Chart (Half Precision) |

### Section Sets (used during profiling with `--set`)
| Set | Sections | Est. Metrics |
|-----|----------|-------------|
| `basic` | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | 213 |
| `detailed` | basic + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SourceCounters, Roofline | 906 |
| `full` | All sections including Instruction/Scheduler/WarpState stats, all Rooflines | 7794 |

### Commonly Used Raw Metrics
| Metric | Description |
|--------|-------------|
| `gpu__time_duration.sum` | Kernel wall-clock duration |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM throughput % |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM throughput % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Active warps % |
| `launch__occupancy_limit_registers` | Occupancy limiter: registers |
| `launch__occupancy_limit_shared_mem` | Occupancy limiter: shared memory |
| `launch__occupancy_limit_blocks` | Occupancy limiter: blocks |
| `launch__occupancy_limit_warps` | Occupancy limiter: warps |
| `sm__sass_thread_inst_executed_op_*` | Per-opcode instruction counts |
| `l1tex__t_sector_hit_rate.pct` | L1 cache hit rate |
| `lts__t_sector_hit_rate.pct` | L2 cache hit rate |

### Available Rules (used during profiling with `--rule`)
| Rule ID | Description |
|---------|-------------|
| `SOLBottleneck` | High-level bottleneck detection |
| `SOLFPRoofline` | Floating Point Roofline Analysis |
| `CPIStall` | Warp stall analysis |
| `Occupancy` | Achieved Occupancy analysis |
| `LaunchConfiguration` | Kernel launch config analysis |
| `HighPipeUtilization` | High pipe utilization bottleneck |
| `IssueSlotUtilization` | Scheduler issue analysis |
| `SharedMemoryConflicts` | Shared memory bank conflicts |
| `ThreadDivergence` | Warp/thread divergence |
| `UncoalescedGlobalAccess` | Uncoalesced global memory |
| `UncoalescedSharedAccess` | Uncoalesced shared memory |
| `SlowPipeLimiter` | Slow pipe limiting compute |
| `FPInstructions` | FP instruction analysis |
| `PCSamplingData` | PC sampling data |

## Comparing Kernels

When the report contains multiple kernels (e.g., a reference nvjet kernel and a tilus kernel), always present metrics side-by-side for comparison. Highlight:
1. Duration difference (which is faster, by how much)
2. Throughput differences (compute vs memory bound)
3. Occupancy differences
4. Any rule findings that differ

## Tips
- The `raw` page has one row per kernel with all metrics as columns — good for extracting specific values.
- The `details` page organizes metrics by section — good for browsing all metrics in a section.
- The `source` page is per-instruction — good for hotspot analysis. Output can be very large; pipe through `head` or filter with `grep`.
- Use `--print-units base` with `--csv` for consistent numeric parsing.
- Use `--print-metric-name name` to get programmatic metric names instead of human labels.
- Source analysis with `cuda,sass` view shows CUDA source lines interleaved with their SASS instructions — extremely useful for correlating high-level code with assembly hotspots.
