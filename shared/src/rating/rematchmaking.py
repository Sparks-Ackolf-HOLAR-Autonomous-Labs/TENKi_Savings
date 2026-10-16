import random
from typing import List, Tuple, Set

class RematchmakingManager:
    """Manages the selection of rematch tasks between agents."""

    def __init__(self):
        self.rematched_pairs_this_checkpoint: Set[Tuple[str, str]] = set()
        self.rematched_pairs_last_checkpoint: Set[Tuple[str, str]] = set()

    def get_rematch_tasks(self, agent_ids: List[str], num_tasks: int) -> List[Tuple[str, str]]:
        """Selects pairs of agents for rematches.

        Args:
            agent_ids: A list of agent IDs to choose from.
            num_tasks: The number of rematch tasks to generate.

        Returns:
            A list of tuples, where each tuple contains two agent IDs for a rematch.
        """
        if len(agent_ids) < 2:
            return []

        tasks = []
        available_pairs = self._get_available_pairs(agent_ids)
        
        for _ in range(num_tasks):
            if not available_pairs:
                break
            pair = random.choice(list(available_pairs))
            tasks.append(pair)
            self.rematched_pairs_this_checkpoint.add(pair)
            available_pairs.remove(pair)
        
        return tasks

    def _get_available_pairs(self, agent_ids: List[str]) -> Set[Tuple[str, str]]:
        """Returns a set of all possible pairs of agents that have not been rematched in the current or previous checkpoint."""
        all_pairs = set()
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                all_pairs.add(tuple(sorted((agent_ids[i], agent_ids[j]))))
        
        return all_pairs - self.rematched_pairs_this_checkpoint - self.rematched_pairs_last_checkpoint

    def end_of_checkpoint(self):
        """Called at the end of a checkpoint to update the rematch history."""
        self.rematched_pairs_last_checkpoint = self.rematched_pairs_this_checkpoint
        self.rematched_pairs_this_checkpoint = set()