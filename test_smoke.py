"""
Quick smoke test for the MIPS Scheduler Environment core components.
Run from the OpenEnv root: python -m pytest envs/mips_scheduler_env/test_smoke.py -v
Or directly: cd OpenEnv && python envs/mips_scheduler_env/test_smoke.py
"""

import sys
import os

# Add paths so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mips_scheduler_env.server.pipeline_simulator import (
    MIPSInstruction, InstructionType, PipelineSimulator, compute_max_possible_stalls
)
from mips_scheduler_env.server.dag_generator import (
    parse_assembly, build_dependency_dag, ScheduleTracker
)
from mips_scheduler_env.server.tasks import get_task, list_tasks, TASKS
from mips_scheduler_env.server.graders import compute_grade


def test_parse_easy_assembly():
    """Test that the easy task's assembly parses correctly."""
    task = get_task("easy_alu_chain")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    print(f"\n=== Easy Task: {len(insts)} instructions parsed ===")
    for inst in insts:
        print(f"  [{inst.id:2d}] {inst.raw_text:30s}  dest={inst.dest_reg}  src={inst.src_regs}")
    assert len(insts) == 10, f"Expected 10 instructions, got {len(insts)}"
    print("  ✓ Parsing OK")


def test_parse_medium_assembly():
    """Test that the medium task's assembly parses correctly."""
    task = get_task("medium_memory_mix")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    print(f"\n=== Medium Task: {len(insts)} instructions parsed ===")
    for inst in insts:
        print(f"  [{inst.id:2d}] {inst.raw_text:30s}  dest={inst.dest_reg}  src={inst.src_regs}")
    assert len(insts) == 20, f"Expected 20 instructions, got {len(insts)}"
    print("  ✓ Parsing OK")


def test_parse_hard_assembly():
    """Test that the hard task's assembly parses correctly."""
    task = get_task("hard_full_block")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    print(f"\n=== Hard Task: {len(insts)} instructions parsed ===")
    for inst in insts:
        print(f"  [{inst.id:2d}] {inst.raw_text:30s}  dest={inst.dest_reg}  src={inst.src_regs}")
    assert len(insts) == 58, f"Expected 58 instructions, got {len(insts)}"
    print("  ✓ Parsing OK")


def test_dependency_dag():
    """Test that the DAG builder identifies correct dependencies."""
    task = get_task("easy_alu_chain")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    dag = build_dependency_dag(insts)
    
    print(f"\n=== DAG for easy_alu_chain ===")
    print(f"  Nodes: {dag.num_nodes}")
    print(f"  Edges: {len(dag.edges)}")
    for e in dag.edges:
        print(f"    {e.from_id} → {e.to_id}  ({e.dep_type}: {e.register})")
    
    # Verify: add $t2, $t0, $t1 (id=2) depends on addi $t0 (id=0) and addi $t1 (id=1)
    edge_pairs = {(e.from_id, e.to_id) for e in dag.edges}
    assert (0, 2) in edge_pairs, "Missing RAW dep: $t0 (0→2)"
    assert (1, 2) in edge_pairs, "Missing RAW dep: $t1 (1→2)"
    print("  ✓ Dependencies OK")


def test_kahn_legal_actions():
    """Test that Kahn's algorithm correctly identifies legal actions."""
    task = get_task("easy_alu_chain")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    dag = build_dependency_dag(insts)
    tracker = ScheduleTracker(dag)
    
    legal = tracker.legal_actions
    print(f"\n=== Kahn's algorithm ===")
    print(f"  Initial legal actions: {legal}")
    
    # Instructions 0 and 1 (addi $t0, addi $t1) should be legal initially
    assert 0 in legal, "Instruction 0 should be legal initially"
    assert 1 in legal, "Instruction 1 should be legal initially"
    # Instruction 2 (add $t2, $t0, $t1) should NOT be legal (depends on 0, 1)
    assert 2 not in legal, "Instruction 2 should not be legal initially"
    
    # Schedule instruction 0
    tracker.schedule(0)
    legal_after = tracker.legal_actions
    print(f"  After scheduling 0: {legal_after}")
    assert 1 in legal_after, "Instruction 1 should still be legal"
    
    # Schedule instruction 1
    tracker.schedule(1)
    legal_after_2 = tracker.legal_actions
    print(f"  After scheduling 0, 1: {legal_after_2}")
    assert 2 in legal_after_2, "Instruction 2 should now be legal"
    
    print("  ✓ Kahn's algorithm OK")


def test_pipeline_simulation():
    """Test pipeline simulation on a known sequence."""
    task = get_task("easy_alu_chain")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    
    sim = PipelineSimulator()
    result = sim.simulate(insts)
    
    print(f"\n=== Pipeline Simulation (easy, original order) ===")
    print(f"  Total cycles: {result.total_cycles}")
    print(f"  Total stalls: {result.total_stalls}")
    print(f"  Stall breakdown: {result.stall_breakdown}")
    
    assert result.total_cycles > 0, "Total cycles should be positive"
    print("  ✓ Simulation OK")


def test_pipeline_simulation_medium():
    """Test pipeline simulation on medium task (with loads + branch)."""
    task = get_task("medium_memory_mix")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    
    sim = PipelineSimulator()
    result = sim.simulate(insts)
    
    print(f"\n=== Pipeline Simulation (medium, original order) ===")
    print(f"  Total cycles: {result.total_cycles}")
    print(f"  Total stalls: {result.total_stalls}")
    print(f"  Stall breakdown: {result.stall_breakdown}")
    
    # Verify load-use stalls exist (lw followed immediately by dependent instruction)
    assert result.total_stalls > 0, "Medium task should have stalls in naive order"
    print("  ✓ Simulation with load-use hazards OK")


def test_grading():
    """Test the grading function on a naive schedule."""
    task = get_task("easy_alu_chain")
    insts = parse_assembly(task.assembly, task.branch_taken_map)
    
    # Naive schedule: original order
    naive_ids = [inst.id for inst in insts]
    grade = compute_grade("easy_alu_chain", naive_ids)
    
    print(f"\n=== Grading ===")
    print(f"  Naive order grade: {grade:.4f}")
    assert 0.0 <= grade <= 1.0, f"Grade should be in [0,1], got {grade}"
    print("  ✓ Grading OK")


def test_full_episode():
    """Test a complete episode: reset → step through all instructions."""
    from mips_scheduler_env.server.mips_scheduler_env import MIPSSchedulerEnvironment
    from mips_scheduler_env.models import SchedulerAction
    
    env = MIPSSchedulerEnvironment()
    
    # Reset
    obs = env.reset(task_name="easy_alu_chain")
    print(f"\n=== Full Episode (easy_alu_chain) ===")
    print(f"  Reset: {obs.num_instructions} instructions, legal={obs.legal_actions}")
    
    step = 0
    while not obs.done:
        # Pick first legal action (naive policy)
        action = SchedulerAction(instruction_id=obs.legal_actions[0])
        obs = env.step(action)
        step += 1
        print(f"  Step {step}: scheduled {action.instruzction_id}, "
              f"stalls={obs.stalls_this_step}, total_stalls={obs.total_stalls}, "
              f"reward={obs.reward}, done={obs.done}")
    
    print(f"  Final grade: {obs.metadata.get('final_grade', 'N/A')}")
    print(f"  Total stalls: {obs.total_stalls}")
    print(f"  Total cycles: {obs.total_cycles}")
    assert obs.done, "Episode should be done"
    assert "final_grade" in obs.metadata, "Should have final_grade"
    grade = obs.metadata["final_grade"]
    assert 0.0 <= grade <= 1.0, f"Grade {grade} not in [0,1]"
    print(f"  ✓ Full episode OK (grade={grade:.4f})")


def test_all_tasks():
    """Run full episodes on all 3 tasks."""
    from mips_scheduler_env.server.mips_scheduler_env import MIPSSchedulerEnvironment
    from mips_scheduler_env.models import SchedulerAction
    
    print(f"\n{'='*60}")
    print(f"  ALL TASKS — Full Episode Test")
    print(f"{'='*60}")
    
    for task_name in list_tasks():
        env = MIPSSchedulerEnvironment()
        obs = env.reset(task_name=task_name)
        
        step = 0
        while not obs.done:
            action = SchedulerAction(instruction_id=obs.legal_actions[0])
            obs = env.step(action)
            step += 1
        
        grade = obs.metadata.get("final_grade", 0.0)
        print(f"  {task_name:25s}  steps={step:3d}  stalls={obs.total_stalls:3d}  "
              f"cycles={obs.total_cycles:3d}  grade={grade:.4f}")
    
    print("  ✓ All tasks completed")


if __name__ == "__main__":
    test_parse_easy_assembly()
    test_parse_medium_assembly()
    test_parse_hard_assembly()
    test_dependency_dag()
    test_kahn_legal_actions()
    test_pipeline_simulation()
    test_pipeline_simulation_medium()
    test_grading()
    test_full_episode()
    test_all_tasks()
    print(f"\n{'='*60}")
    print(f"  ALL SMOKE TESTS PASSED ✓")
    print(f"{'='*60}")
