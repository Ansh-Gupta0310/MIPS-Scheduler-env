"""
Task Definitions for MIPS Instruction Scheduling
==================================================
Three difficulty tiers, each containing hand-crafted MIPS assembly code
that represents realistic basic blocks with varying complexity.

Each task defines:
  - Assembly code (the basic block to schedule)
  - Branch-taken map (which branches are taken at runtime for prediction scoring)
  - Difficulty metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskConfig:
    """Configuration for a single scheduling task."""
    name: str
    difficulty: str          # "easy", "medium", "hard"
    assembly: str            # Multi-line MIPS assembly code
    branch_taken_map: Dict[int, bool] = field(default_factory=dict)
    description: str = ""
    max_steps: int = 50      # Maximum steps before episode is truncated


# ════════════════════════════════════════════════════════════════════════════
# EASY TASK: 10 instructions, R-type + addi only, no memory, no branches
# ════════════════════════════════════════════════════════════════════════════

EASY_ASSEMBLY = """\
addi $t0, $zero, 10
addi $t1, $zero, 20
add  $t2, $t0, $t1
sub  $t3, $t0, $t1
and  $t4, $t2, $t3
or   $t5, $t2, $t3
slt  $t6, $t4, $t5
add  $t7, $t4, $t5
sub  $t8, $t6, $t7
add  $t9, $t8, $t0
"""

# ════════════════════════════════════════════════════════════════════════════
# MEDIUM TASK: 20 instructions, R + I + lw/sw + branch
# Simulates a partial loop body computing array sum with conditional check
# ════════════════════════════════════════════════════════════════════════════

MEDIUM_ASSEMBLY = """\
addi $t0, $zero, 0
addi $t1, $zero, 100
lw   $t2, 0($s0)
lw   $t3, 4($s0)
add  $t4, $t2, $t3
lw   $t5, 8($s0)
add  $t4, $t4, $t5
sw   $t4, 0($s1)
lw   $t6, 12($s0)
lw   $t7, 16($s0)
sub  $t8, $t6, $t7
slt  $s2, $t8, $zero
add  $s3, $t6, $t7
sw   $s3, 4($s1)
addi $s0, $s0, 20
addi $s1, $s1, 8
sub  $t9, $t1, $s0
slt  $s4, $zero, $t9
or   $s5, $s2, $s4
beq  $s5, $zero, loop_end
"""

# ════════════════════════════════════════════════════════════════════════════
# HARD TASK: 40 instructions, full set with multiple loads, stores, branches
# Simulates: matrix row operation with conditional accumulation
# ════════════════════════════════════════════════════════════════════════════
HARD_ASSEMBLY = """\
addi $s0, $zero, 0
addi $s1, $zero, 1000
addi $t0, $zero, 10
add  $t2, $zero, $zero
add  $t3, $zero, $zero
loop:
lw   $t1, 0($s0)
add  $t2, $t1, $t1
lw   $t3, 4($s0)
sw   $t3, 0($s1)
lw   $t4, 8($s0)
sub  $t5, $t4, $t2
sw   $t5, 4($s1)
lw   $t6, 12($s0)
or   $t7, $t6, $t5
sw   $t7, 8($s1)
lw   $t8, 16($s0)
and  $t9, $t8, $t7
sw   $t9, 12($s1)
lw   $s2, 20($s0)
xor  $s3, $s2, $t9
sw   $s3, 16($s1)
lw   $s4, 24($s0)
slt  $s5, $s4, $s3
sw   $s5, 20($s1)
lw   $s6, 28($s0)
sll  $s7, $s6, 2
sw   $s7, 24($s1)
lw   $k0, 32($s0)
srl  $k1, $k0, 2
add  $v0, $s7, $k1
sw   $v0, 28($s1)
lw   $v1, 36($s0)
sub  $a0, $v1, $v0
sw   $a0, 32($s1)
lw   $a1, 40($s0)
add  $a2, $a1, $a0
sw   $a2, 36($s1)
lw   $a3, 44($s0)
sub  $t1, $a3, $a2
sw   $t1, 40($s1)
lw   $t2, 48($s0)
or   $t3, $t2, $t1
sw   $t3, 44($s1)
lw   $t4, 52($s0)
and  $t5, $t4, $t3
sw   $t5, 48($s1)
lw   $t6, 56($s0)
xor  $t7, $t6, $t5
sw   $t7, 52($s1)
lw   $t8, 60($s0)
add  $t9, $t8, $t7
sw   $t9, 56($s1)
addi $s0, $s0, 64
addi $s1, $s1, 60
addi $t0, $t0, -1
bne  $t0, $zero, loop
nop
done:
add  $zero, $zero, $zero
"""

# ════════════════════════════════════════════════════════════════════════════
# Task Registry
# ════════════════════════════════════════════════════════════════════════════

TASKS: Dict[str, TaskConfig] = {
    "easy_alu_chain": TaskConfig(
        name="easy_alu_chain",
        difficulty="easy",
        assembly=EASY_ASSEMBLY,
        branch_taken_map={},
        description=(
            "Schedule 10 ALU instructions (R-type + addi). "
            "No memory operations or branches. Focus on minimizing "
            "RAW data hazard stalls through instruction reordering."
        ),
        max_steps=15,
    ),
    "medium_memory_mix": TaskConfig(
        name="medium_memory_mix",
        difficulty="medium",
        assembly=MEDIUM_ASSEMBLY,
        branch_taken_map={19: True},  # beq at index 19 is taken (loop continues)
        description=(
            "Schedule 20 instructions including loads, stores, ALU ops, "
            "and a conditional branch. Must handle load-use hazards and "
            "branch misprediction penalties. Optimal scheduling separates "
            "loads from their dependent ALU consumers."
        ),
        max_steps=30,
    ),
    "hard_full_block": TaskConfig(
        name="hard_full_block",
        difficulty="hard",
        assembly=HARD_ASSEMBLY,
        branch_taken_map={39: True},  # beq at the end is taken (loop continues)
        description=(
            "Schedule 40 instructions from a realistic matrix operation. "
            "Contains many loads creating load-use hazards, stores creating "
            "memory ordering constraints, complex dependency chains, and "
            "a conditional branch. Requires deep understanding of pipeline "
            "microarchitecture to schedule optimally."
        ),
        max_steps=60,
    ),
}


def get_task(name: str) -> TaskConfig:
    """Get a task configuration by name."""
    if name not in TASKS:
        available = ", ".join(TASKS.keys())
        raise ValueError(f"Unknown task: {name!r}. Available tasks: {available}")
    return TASKS[name]


def list_tasks() -> List[str]:
    """Return list of available task names."""
    return list(TASKS.keys())
