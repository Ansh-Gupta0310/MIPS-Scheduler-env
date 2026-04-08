"""
DAG Generator — Assembly Parser & Dependency Analyzer
======================================================
1. Parses MIPS assembly text into structured MIPSInstruction objects.
2. Performs dependency analysis (RAW, WAR, WAW) to build a DAG.
3. Provides Kahn's-algorithm-based legal-action tracking for the RL agent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .pipeline_simulator import InstructionType, MIPSInstruction


# ─── Assembly Parser ───────────────────────────────────────────────────────

# Register aliases
_REG_ALIASES = {
    "zero": "$0",
    "at": "$1",
    "v0": "$2", "v1": "$3",
    "a0": "$4", "a1": "$5", "a2": "$6", "a3": "$7",
    "t0": "$8", "t1": "$9", "t2": "$10", "t3": "$11",
    "t4": "$12", "t5": "$13", "t6": "$14", "t7": "$15",
    "s0": "$16", "s1": "$17", "s2": "$18", "s3": "$19",
    "s4": "$20", "s5": "$21", "s6": "$22", "s7": "$23",
    "t8": "$24", "t9": "$25",
    "k0": "$26", "k1": "$27",
    "gp": "$28", "sp": "$29", "fp": "$30", "ra": "$31",
}


def _normalize_reg(reg: str) -> str:
    """Normalize register name to canonical $N form."""
    reg = reg.strip().replace(" ", "")
    if reg.startswith("$"):
        name = reg[1:]
        if name in _REG_ALIASES:
            return _REG_ALIASES[name]
        return reg  # already $0..$31
    return reg


# Opcode classification
_R_TYPE_OPS = {"add", "sub", "and", "or", "xor", "nor", "slt", "sll", "srl", "sra",
               "addu", "subu", "sltu", "mult", "div", "mfhi", "mflo", "jr"}
_I_TYPE_OPS = {"addi", "addiu", "andi", "ori", "xori", "slti", "sltiu",
               "lw", "sw", "lb", "sb", "lh", "sh", "lui",
               "beq", "bne", "bgtz", "blez"}
_J_TYPE_OPS = {"j", "jal"}


def _classify_opcode(opcode: str) -> InstructionType:
    op = opcode.lower()
    if op in _R_TYPE_OPS:
        return InstructionType.R_TYPE
    if op in _I_TYPE_OPS:
        return InstructionType.I_TYPE
    if op in _J_TYPE_OPS:
        return InstructionType.J_TYPE
    if op == "nop":
        return InstructionType.NOP
    return InstructionType.R_TYPE  # default fallback


def _parse_memory_operand(operand: str) -> Tuple[str, str]:
    """Parse 'offset($reg)' → (offset, $reg)."""
    m = re.match(r"(-?\d+)\((\$\w+)\)", operand.strip())
    if m:
        return m.group(1), _normalize_reg(m.group(2))
    # might be just ($reg) with implicit 0
    m = re.match(r"\((\$\w+)\)", operand.strip())
    if m:
        return "0", _normalize_reg(m.group(1))
    return "0", operand.strip()


def parse_instruction(line: str, inst_id: int, branch_taken: bool = False) -> Optional[MIPSInstruction]:
    """
    Parse a single MIPS assembly line into a MIPSInstruction.
    Returns None for empty lines, comments, and labels.
    """
    # Strip comments and whitespace
    line = line.split("#")[0].strip()
    if not line or line.endswith(":"):
        return None

    # Remove label prefix if present (e.g., "loop: add $t0, $t1, $t2")
    if ":" in line:
        line = line.split(":", 1)[1].strip()
    if not line:
        return None

    parts = line.replace(",", " ").split()
    if not parts:
        return None

    opcode = parts[0].lower()
    operands = [p.strip() for p in parts[1:] if p.strip()]

    inst_type = _classify_opcode(opcode)
    dest_reg: Optional[str] = None
    src_regs: List[str] = []

    if opcode == "nop":
        pass

    # R-type: add $d, $s, $t
    elif opcode in ("add", "sub", "and", "or", "xor", "nor", "slt",
                     "addu", "subu", "sltu"):
        if len(operands) >= 3:
            dest_reg = _normalize_reg(operands[0])
            src_regs = [_normalize_reg(operands[1]), _normalize_reg(operands[2])]

    # Shift: sll $d, $t, shamt
    elif opcode in ("sll", "srl", "sra"):
        if len(operands) >= 2:
            dest_reg = _normalize_reg(operands[0])
            src_regs = [_normalize_reg(operands[1])]

    # Move from hi/lo
    elif opcode in ("mfhi", "mflo"):
        if operands:
            dest_reg = _normalize_reg(operands[0])

    # jr $ra
    elif opcode == "jr":
        if operands:
            src_regs = [_normalize_reg(operands[0])]

    # I-type ALU: addi $t, $s, imm
    elif opcode in ("addi", "addiu", "andi", "ori", "xori", "slti", "sltiu"):
        if len(operands) >= 2:
            dest_reg = _normalize_reg(operands[0])
            src_regs = [_normalize_reg(operands[1])]

    # lui $t, imm
    elif opcode == "lui":
        if operands:
            dest_reg = _normalize_reg(operands[0])

    # Load: lw $t, offset($s)
    elif opcode in ("lw", "lb", "lh"):
        if len(operands) >= 2:
            dest_reg = _normalize_reg(operands[0])
            _, base_reg = _parse_memory_operand(operands[1])
            src_regs = [base_reg]

    # Store: sw $t, offset($s)  — $t is source, base reg is source, no dest
    elif opcode in ("sw", "sb", "sh"):
        if len(operands) >= 2:
            src_regs = [_normalize_reg(operands[0])]
            _, base_reg = _parse_memory_operand(operands[1])
            src_regs.append(base_reg)

    # Branch: beq $s, $t, label
    elif opcode in ("beq", "bne"):
        if len(operands) >= 2:
            src_regs = [_normalize_reg(operands[0]), _normalize_reg(operands[1])]

    # Branch: bgtz $s, label
    elif opcode in ("bgtz", "blez"):
        if operands:
            src_regs = [_normalize_reg(operands[0])]

    # Jump: j label (no register operands)
    elif opcode == "j":
        pass  # no register deps

    # jal label → writes $ra
    elif opcode == "jal":
        dest_reg = "$31"  # $ra

    # Filter out $0 / $zero from sources (hardwired zero)
    src_regs = [r for r in src_regs if r not in ("$0", "$zero")]

    return MIPSInstruction(
        id=inst_id,
        opcode=opcode,
        dest_reg=dest_reg,
        src_regs=tuple(src_regs),
        inst_type=inst_type,
        raw_text=line.strip(),
        branch_taken=branch_taken,
    )


def parse_assembly(assembly_text: str, branch_taken_map: Optional[Dict[int, bool]] = None) -> List[MIPSInstruction]:
    """
    Parse a multi-line MIPS assembly program into a list of MIPSInstruction.
    
    Args:
        assembly_text: Multi-line MIPS assembly string
        branch_taken_map: Dict mapping instruction index → whether branch is taken
                         at runtime (for branch prediction scoring)
    """
    if branch_taken_map is None:
        branch_taken_map = {}

    instructions: List[MIPSInstruction] = []
    idx = 0
    for line in assembly_text.strip().splitlines():
        taken = branch_taken_map.get(idx, False)
        inst = parse_instruction(line, inst_id=idx, branch_taken=taken)
        if inst is not None:
            instructions.append(inst)
            idx += 1
    return instructions


# ─── Dependency Analysis & DAG ─────────────────────────────────────────────

class DependencyType(str):
    RAW = "RAW"   # Read-After-Write  (true dependency)
    WAR = "WAR"   # Write-After-Read  (anti-dependency)
    WAW = "WAW"   # Write-After-Write (output dependency)


@dataclass
class DependencyEdge:
    """An edge in the instruction dependency graph."""
    from_id: int
    to_id: int
    dep_type: str       # "RAW", "WAR", "WAW"
    register: str       # The register causing the dependency


@dataclass
class InstructionDAG:
    """
    A Directed Acyclic Graph representing instruction dependencies.
    
    Nodes are MIPSInstruction objects, edges are DependencyEdge objects.
    """
    instructions: List[MIPSInstruction]
    edges: List[DependencyEdge]
    # Adjacency list: node_id → list of successor node_ids
    successors: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    # Reverse adjacency: node_id → list of predecessor node_ids
    predecessors: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        # Rebuild adjacency lists from edges
        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        for e in self.edges:
            if e.to_id not in self.successors[e.from_id]:
                self.successors[e.from_id].append(e.to_id)
            if e.from_id not in self.predecessors[e.to_id]:
                self.predecessors[e.to_id].append(e.from_id)

    @property
    def num_nodes(self) -> int:
        return len(self.instructions)

    def instruction_by_id(self, inst_id: int) -> MIPSInstruction:
        for inst in self.instructions:
            if inst.id == inst_id:
                return inst
        raise KeyError(f"No instruction with id={inst_id}")

    def to_dict(self) -> dict:
        """Serialize for observation."""
        return {
            "instructions": [
                {
                    "id": inst.id,
                    "opcode": inst.opcode,
                    "dest_reg": inst.dest_reg,
                    "src_regs": list(inst.src_regs),
                    "type": inst.inst_type.value,
                    "raw_text": inst.raw_text,
                }
                for inst in self.instructions
            ],
            "edges": [
                [e.from_id, e.to_id, e.dep_type, e.register]
                for e in self.edges
            ],
        }


def build_dependency_dag(instructions: List[MIPSInstruction]) -> InstructionDAG:
    """
    Build a dependency DAG from an ordered list of instructions.
    
    Identifies three types of dependencies:
      - RAW (Read-After-Write): Inst B reads a register that Inst A writes
      - WAR (Write-After-Read): Inst B writes a register that Inst A reads
      - WAW (Write-After-Write): Inst B writes the same register as Inst A
    
    Edges go from the earlier instruction to the later instruction,
    meaning the later instruction depends on the earlier one.
    """
    edges: List[DependencyEdge] = []
    edge_set: Set[Tuple[int, int]] = set()  # deduplicate

    def _add_edge(from_id: int, to_id: int, dep_type: str, register: str):
        key = (from_id, to_id)
        if key not in edge_set:
            edge_set.add(key)
            edges.append(DependencyEdge(from_id, to_id, dep_type, register))

    # For each instruction, find dependencies with all earlier instructions
    for j in range(len(instructions)):
        inst_j = instructions[j]

        for i in range(j):
            inst_i = instructions[i]

            # RAW: inst_i writes reg, inst_j reads reg
            if inst_i.writes_reg and inst_i.dest_reg:
                for src in inst_j.src_regs:
                    if src == inst_i.dest_reg:
                        # Check no intermediate writer (we want the closest producer)
                        # But for DAG correctness, we add ALL RAW edges.
                        # The transitive reduction happens naturally in Kahn's alg.
                        _add_edge(inst_i.id, inst_j.id, DependencyType.RAW, src)

            # WAR: inst_j writes reg, inst_i reads reg
            if inst_j.writes_reg and inst_j.dest_reg:
                for src in inst_i.src_regs:
                    if src == inst_j.dest_reg:
                        _add_edge(inst_i.id, inst_j.id, DependencyType.WAR, src)

            # WAW: both write the same register
            if (inst_i.writes_reg and inst_j.writes_reg
                    and inst_i.dest_reg and inst_j.dest_reg
                    and inst_i.dest_reg == inst_j.dest_reg):
                _add_edge(inst_i.id, inst_j.id, DependencyType.WAW, inst_i.dest_reg)

    # Memory dependencies: sw/lw to the same address
    # Since we don't track addresses precisely, conservatively assume
    # all stores alias with all subsequent loads/stores.
    for i in range(len(instructions)):
        for j in range(i + 1, len(instructions)):
            inst_i = instructions[i]
            inst_j = instructions[j]

            # Store-Load (RAW on memory): store then load → load depends on store
            if inst_i.is_store and inst_j.is_load:
                _add_edge(inst_i.id, inst_j.id, DependencyType.RAW, "MEM")

            # Store-Store (WAW on memory): two stores → order must be preserved
            if inst_i.is_store and inst_j.is_store:
                _add_edge(inst_i.id, inst_j.id, DependencyType.WAW, "MEM")

            # Load-Store (WAR on memory): load then store → store must come after
            if inst_i.is_load and inst_j.is_store:
                _add_edge(inst_i.id, inst_j.id, DependencyType.WAR, "MEM")

    return InstructionDAG(instructions=instructions, edges=edges)


# ─── Kahn's Algorithm — Legal Action Tracker ───────────────────────────────

class ScheduleTracker:
    """
    Tracks which instructions have been scheduled and which are currently
    legal to schedule (in-degree == 0 among unscheduled nodes).
    
    Implements Kahn's topological sort with the agent choosing the order.
    """

    def __init__(self, dag: InstructionDAG):
        self.dag = dag
        self._scheduled: Set[int] = set()
        # Build in-degree counts (only among unscheduled nodes)
        self._in_degree: Dict[int, int] = {}
        for inst in dag.instructions:
            self._in_degree[inst.id] = len(dag.predecessors.get(inst.id, []))

    @property
    def scheduled_ids(self) -> List[int]:
        return sorted(self._scheduled)

    @property
    def legal_actions(self) -> List[int]:
        """Return IDs of instructions with all predecessors scheduled."""
        result = []
        for inst in self.dag.instructions:
            if inst.id in self._scheduled:
                continue
            if self._in_degree[inst.id] == 0:
                result.append(inst.id)
        return sorted(result)

    @property
    def is_complete(self) -> bool:
        return len(self._scheduled) == self.dag.num_nodes

    def schedule(self, inst_id: int) -> None:
        """
        Mark an instruction as scheduled and update in-degrees of its successors.
        
        Raises ValueError if inst_id is not a legal action.
        """
        if inst_id in self._scheduled:
            raise ValueError(f"Instruction {inst_id} already scheduled")
        if self._in_degree.get(inst_id, 0) != 0:
            raise ValueError(
                f"Instruction {inst_id} still has {self._in_degree[inst_id]} "
                f"unscheduled predecessors — not a legal action"
            )

        self._scheduled.add(inst_id)

        # Decrement in-degree of all successors
        for succ_id in self.dag.successors.get(inst_id, []):
            if succ_id not in self._scheduled:
                self._in_degree[succ_id] -= 1

    def get_scheduled_instructions(self) -> List[MIPSInstruction]:
        """Return the list of scheduled instructions in scheduling order."""
        # We need to track order, not just the set
        return [self.dag.instruction_by_id(i) for i in self.scheduled_ids]
