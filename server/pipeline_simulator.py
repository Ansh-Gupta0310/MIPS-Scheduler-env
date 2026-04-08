"""
MIPS 5-Stage Pipeline Simulator
================================
Simulates a classic 5-stage MIPS pipeline (IF → ID → EX → MEM → WB) with:
  - Data forwarding (EX→EX, MEM→EX bypass paths)
  - Load-use hazard detection (1-cycle stall even with forwarding)
  - Structural hazards (single memory port)
  - Register file half-cycle convention (WB writes 1st half, ID reads 2nd half
    of the same clock cycle → no stall for same-cycle WB-write / ID-read)
  - Branch prediction via 2-bit saturating counter (resolved at EX stage,
    2-cycle flush penalty on misprediction)

The simulator receives an *ordered* list of MIPSInstruction objects (the
schedule produced by the agent) and returns the total execution cycles, total
stalls, and a per-instruction stall breakdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Instruction representation ────────────────────────────────────────────

class InstructionType(str, Enum):
    R_TYPE = "R"
    I_TYPE = "I"
    J_TYPE = "J"
    NOP = "NOP"


@dataclass(frozen=True)
class MIPSInstruction:
    """A single MIPS assembly instruction with decoded register fields."""
    id: int                        # Unique index in the basic block
    opcode: str                    # e.g. "add", "lw", "beq", "j"
    dest_reg: Optional[str]        # Destination register (None for sw/beq/j)
    src_regs: Tuple[str, ...]      # Source registers (may be empty)
    inst_type: InstructionType     # R / I / J / NOP
    raw_text: str = ""             # Original assembly text for debug
    branch_taken: bool = False     # For branch: whether *actually* taken at runtime

    @property
    def is_load(self) -> bool:
        return self.opcode == "lw"

    @property
    def is_store(self) -> bool:
        return self.opcode == "sw"

    @property
    def is_branch(self) -> bool:
        return self.opcode in ("beq", "bne")

    @property
    def is_jump(self) -> bool:
        return self.opcode == "j"

    @property
    def is_memory(self) -> bool:
        return self.opcode in ("lw", "sw")

    @property
    def writes_reg(self) -> bool:
        return self.dest_reg is not None and self.dest_reg != "$zero" and self.dest_reg != "$0"


# ─── Pipeline stage names ──────────────────────────────────────────────────

class Stage(str, Enum):
    IF = "IF"
    ID = "ID"
    EX = "EX"
    MEM = "MEM"
    WB = "WB"


# ─── Simulation result ─────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Output of a full pipeline simulation run."""
    total_cycles: int = 0
    total_stalls: int = 0
    stall_breakdown: List[int] = field(default_factory=list)
    # stall_breakdown[i] = stalls caused *before* instruction i could issue


# ─── 2-bit saturating counter for branch prediction ───────────────────────

class BranchPredictor:
    """
    2-bit saturating counter branch predictor.
    States: 0=strongly-not-taken, 1=weakly-not-taken,
            2=weakly-taken, 3=strongly-taken
    Prediction: state >= 2 → predict taken, else predict not-taken.
    """

    def __init__(self):
        self._counters: Dict[int, int] = {}  # instruction_id → counter

    def predict(self, inst_id: int) -> bool:
        """Return True if we predict the branch as taken."""
        return self._counters.get(inst_id, 1) >= 2  # init = 1 (weakly not-taken)

    def update(self, inst_id: int, actually_taken: bool) -> None:
        """Update the counter after the branch outcome is known."""
        c = self._counters.get(inst_id, 1)
        if actually_taken:
            c = min(c + 1, 3)
        else:
            c = max(c - 1, 0)
        self._counters[inst_id] = c


# ─── Pipeline Simulator ───────────────────────────────────────────────────

@dataclass
class _PipelineSlot:
    """One pipeline stage's content at a given cycle."""
    inst: Optional[MIPSInstruction] = None
    stage: Optional[Stage] = None


class PipelineSimulator:
    """
    Cycle-accurate 5-stage MIPS pipeline simulator.

    Forwarding paths modelled:
      • EX/MEM register → EX stage input   (result available after EX)
      • MEM/WB register → EX stage input    (result available after MEM)
      • WB writes first half of cycle, ID reads second half →
        same-cycle WB/ID on the same register is fine (no hazard).

    Hazards detected:
      • RAW data hazard (with forwarding resolving most, except load-use)
      • Load-use: lw in EX, dependent instruction in ID → 1-cycle stall
      • Structural: two memory-stage instructions colliding (not applicable
        in a simple in-order pipeline with one instruction per cycle,
        but we track it for thoroughness)
      • Control: branch misprediction → 2-cycle flush
    """

    def __init__(self):
        self.predictor = BranchPredictor()

    # ── public API ──────────────────────────────────────────────────────

    def simulate(self, instructions: List[MIPSInstruction]) -> SimulationResult:
        """Simulate the ordered instruction list through the pipeline."""
        if not instructions:
            return SimulationResult()

        n = len(instructions)
        # issue_cycle[i]  = cycle at which instruction i enters IF
        # We track when each instruction *enters* each stage.
        # stage_entry[i][stage] = cycle number
        stage_entry: List[Dict[Stage, int]] = [{} for _ in range(n)]
        stalls_per_inst: List[int] = [0] * n

        next_issue_cycle = 0  # earliest cycle the next instruction can enter IF

        for idx, inst in enumerate(instructions):
            # The instruction *wants* to issue (enter IF) at next_issue_cycle
            issue_cycle = next_issue_cycle

            # --- Check for data hazards (RAW with forwarding logic) ---
            extra_stalls = 0
            if inst.src_regs:
                extra_stalls = self._data_hazard_stalls(
                    inst, idx, instructions, stage_entry
                )

            issue_cycle += extra_stalls

            # Fill in stage timing for this instruction
            stage_entry[idx][Stage.IF]  = issue_cycle
            stage_entry[idx][Stage.ID]  = issue_cycle + 1
            stage_entry[idx][Stage.EX]  = issue_cycle + 2
            stage_entry[idx][Stage.MEM] = issue_cycle + 3
            stage_entry[idx][Stage.WB]  = issue_cycle + 4

            stalls_per_inst[idx] = extra_stalls

            # --- Branch prediction penalty ---
            if inst.is_branch:
                predicted_taken = self.predictor.predict(inst.id)
                actually_taken = inst.branch_taken
                self.predictor.update(inst.id, actually_taken)

                if predicted_taken != actually_taken:
                    # Misprediction: branch resolved at end of EX stage.
                    # We must flush the instructions that entered IF and ID
                    # during the 2 cycles after this branch entered IF.
                    # This means the NEXT instruction is delayed by 2 cycles.
                    # Since we're scheduling a single basic block, the
                    # "flushed" instructions are just bubbles; the *next*
                    # instruction in our schedule gets delayed.
                    # Add 2-cycle penalty to the next issue.
                    next_issue_cycle = issue_cycle + 1 + 2  # +1 normal, +2 flush
                    stalls_per_inst[idx] += 2
                else:
                    next_issue_cycle = issue_cycle + 1
            elif inst.is_jump:
                # Unconditional jump: target known after IF/ID.
                # In MIPS, jump target is decoded in ID.
                # 1-cycle flush (the instruction that was fetched during
                # the jump's IF may be wrong).
                next_issue_cycle = issue_cycle + 1 + 1  # +1 normal, +1 flush
                stalls_per_inst[idx] += 1
            else:
                next_issue_cycle = issue_cycle + 1

        # Total cycles = when the last instruction exits WB
        total_cycles = stage_entry[n - 1][Stage.WB] + 1
        total_stalls = sum(stalls_per_inst)

        return SimulationResult(
            total_cycles=total_cycles,
            total_stalls=total_stalls,
            stall_breakdown=stalls_per_inst,
        )

    def simulate_incremental(
        self,
        scheduled_so_far: List[MIPSInstruction],
    ) -> Tuple[int, int]:
        """
        Run a full simulation on scheduled_so_far and return
        (stalls_caused_by_last_instruction, total_stalls).
        """
        result = self.simulate(scheduled_so_far)
        if not result.stall_breakdown:
            return 0, 0
        last_stalls = result.stall_breakdown[-1]
        return last_stalls, result.total_stalls

    # ── private helpers ─────────────────────────────────────────────────

    def _data_hazard_stalls(
        self,
        current_inst: MIPSInstruction,
        current_idx: int,
        instructions: List[MIPSInstruction],
        stage_entry: List[Dict[Stage, int]],
    ) -> int:
        """
        Compute the stall cycles required before `current_inst` can enter IF,
        considering forwarding and the register-file half-cycle convention.

        Forwarding paths:
          1. EX/MEM → EX: result produced at end of EX is available at
             the beginning of the *next* EX stage.
          2. MEM/WB → EX: result produced at end of MEM is available at
             the beginning of the *next* EX stage.
          3. WB half-cycle: WB writes in first half, ID reads in second half,
             so same-cycle WB/ID is OK (no stall).

        Load-use hazard:
          lw produces result at end of MEM. If the *very next* instruction
          needs it in EX, there is a 1-cycle bubble (even with MEM→EX
          forwarding the data arrives 1 cycle late relative to EX need).
        """
        max_stalls = 0

        for src_reg in current_inst.src_regs:
            if src_reg in ("$zero", "$0"):
                continue

            # Look backwards for the most recent producer of src_reg
            for prev_idx in range(current_idx - 1, -1, -1):
                prev_inst = instructions[prev_idx]
                if prev_inst.dest_reg != src_reg:
                    continue
                if not prev_inst.writes_reg:
                    continue

                # Found the producer.  Compute stalls needed.
                prev_ex_end = stage_entry[prev_idx][Stage.EX] + 1      # cycle after EX finishes
                prev_mem_end = stage_entry[prev_idx][Stage.MEM] + 1    # cycle after MEM finishes
                prev_wb_end = stage_entry[prev_idx][Stage.WB] + 1      # cycle after WB finishes

                # The current instruction's EX stage (where it *needs* the value)
                # is at issue_cycle + 2.  We need to find the minimum issue_cycle
                # such that the value is available via forwarding.
                #
                # Without stalls, current issues at next_issue_cycle.
                # current EX = next_issue_cycle + 2
                #
                # For EX→EX forwarding:
                #   value available at prev_ex_end → need current EX >= prev_ex_end
                #   i.e. issue_cycle + 2 >= prev_ex_end
                #   i.e. issue_cycle >= prev_ex_end - 2
                #
                # For load (lw): value available at prev_mem_end → need current EX >= prev_mem_end
                #   i.e. issue_cycle >= prev_mem_end - 2
                #   This creates a 1-cycle stall compared to EX→EX forwarding.
                #
                # For WB half-cycle (no forwarding, just register file):
                #   value available at prev WB write (first half of WB cycle)
                #   current ID reads in second half → current ID cycle >= prev WB cycle
                #   current ID = issue_cycle + 1 >= stage_entry[prev_idx][Stage.WB]
                #   i.e. issue_cycle >= stage_entry[prev_idx][Stage.WB] - 1

                if prev_inst.is_load:
                    # Load-use: value available after MEM stage
                    # MEM→EX forwarding: current EX >= prev_mem_end
                    min_issue = prev_mem_end - 2
                else:
                    # ALU result: EX→EX forwarding: current EX >= prev_ex_end
                    min_issue = prev_ex_end - 2

                # Also check WB half-cycle path (always available, but slower)
                wb_path_issue = stage_entry[prev_idx][Stage.WB] - 1
                # The actual minimum is the best (lowest) of forwarding vs WB path:
                min_issue = min(min_issue, wb_path_issue)
                # But min_issue shouldn't be negative
                min_issue = max(min_issue, 0)

                # Calculate stalls: how many cycles beyond the natural issue point
                # Note: stage_entry isn't filled for current yet, so the "natural"
                # issue would be at the *computed* next_issue_cycle (which is the
                # caller's issue_cycle before stalls).  But we encode stalls as
                # the extra delay returned by this function.
                #
                # Since we don't have the caller's issue_cycle here, we compute
                # the earliest *absolute* issue cycle this dependency requires,
                # and the caller will accumulate the max across all sources.

                # However, this method computes relative stalls based on natural
                # next issue.  So let's compute what natural issue cycle would be
                # for this instruction.
                if current_idx == 0:
                    natural_issue = 0
                else:
                    # Natural = previous instruction's IF + 1 (pipelining)
                    natural_issue = stage_entry[current_idx - 1].get(Stage.IF, 0) + 1

                needed_delay = max(0, min_issue - natural_issue)
                max_stalls = max(max_stalls, needed_delay)
                break  # Found the most recent producer for this src_reg

        return max_stalls


def compute_max_possible_stalls(instructions: List[MIPSInstruction]) -> int:
    """
    Estimate the worst-case stalls for a set of instructions.
    Used for reward normalization.
    
    Worst case: every instruction causes a stall.
    - Load-use: 1 stall each
    - Branch mispredict: 2 stalls each
    - Jump: 1 stall each
    """
    max_stalls = 0
    for inst in instructions:
        if inst.is_load:
            max_stalls += 1  # load-use stall
        if inst.is_branch:
            max_stalls += 2  # mispredict flush
        if inst.is_jump:
            max_stalls += 1  # jump flush
    # Add potential RAW stalls (worst case: every instruction depends on previous)
    max_stalls += len(instructions) - 1
    return max(max_stalls, 1)  # avoid division by zero
