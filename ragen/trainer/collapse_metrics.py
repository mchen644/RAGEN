"""
Collapse detection metrics for RL training.

Distinguishes two collapse phenomena:
- Entropy Collapse: Model becomes more deterministic per input (low H(R|X))
- Template Collapse: Reasoning becomes input-independent (low I(X;R))

Key insight: We compute MI under the batch's empirical input distribution (uniform over prompts),
not the true p(x). This is exactly what's needed for diagnosing template collapse.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto


class CollapseDetector:
    """
    Detects template collapse and entropy collapse in RL training.

    Key metrics:
    - MI Estimate: Î(X;R) - High = healthy, Low = template collapse
    - Retrieval Accuracy: Acc - High = healthy, ~1/N = template collapse
    """

    def __init__(
        self,
        enabled: bool = False,
        compute_freq: int = 10,
        micro_batch_size: int = 16,
        context_window_mode: str = "full",
        multi_turn_enabled: bool = True,
        multi_turn_strategy: str = "turn_uniform",
        num_samples: Optional[int] = None,
    ):
        """
        Initialize the collapse detector.

        Args:
            enabled: Whether collapse detection is enabled
            compute_freq: Compute metrics every N steps
            micro_batch_size: Micro batch size for cross-scoring
            context_window_mode: Context window mode ("full", "single_turn", "limited_multi_turn")
            multi_turn_enabled: Whether to use multi-turn sampling for MI computation
            multi_turn_strategy: Sampling strategy ("turn_uniform" or "trajectory_uniform")
            num_samples: Number of (x,r) pairs to sample (None = use all)
        """
        self.enabled = enabled
        self.compute_freq = compute_freq
        self.micro_batch_size = micro_batch_size
        self.context_window_mode = context_window_mode
        self.multi_turn_enabled = multi_turn_enabled
        self.multi_turn_strategy = multi_turn_strategy
        self.num_samples = num_samples

    def compute_collapse_metrics(
        self,
        batch: DataProto,
        actor_compute_log_prob_fn: Callable[[DataProto], DataProto],
        global_step: int,
    ) -> Dict[str, float]:
        """
        Compute collapse detection metrics.

        Args:
            batch: The training batch with input_ids, attention_mask, position_ids, responses
            actor_compute_log_prob_fn: Function to compute log probabilities
            global_step: Current global training step

        Returns:
            Dictionary of collapse metrics (empty if not enabled or not the right step)
        """
        if not self.enabled:
            return {}

        if global_step % self.compute_freq != 0:
            return {}

        # Only compute collapse metrics when context_window_mode='full'
        # In other modes, same group_id doesn't mean same prompt
        if self.context_window_mode != "full":
            return {}

        # Get group_ids from batch
        if batch.non_tensor_batch is None or "group_ids" not in batch.non_tensor_batch:
            return {}

        group_ids = batch.non_tensor_batch["group_ids"]
        if isinstance(group_ids, list):
            group_ids = np.array(group_ids)

        if batch.non_tensor_batch is None:
            raise ValueError("Collapse metrics require non_tensor_batch with prompt/reasoning data.")

        # Check for multi-turn data
        all_turns_prompt_ids = batch.non_tensor_batch.get("all_turns_prompt_ids")
        all_turns_reasoning_ids = batch.non_tensor_batch.get("all_turns_reasoning_ids")
        turn_counts = batch.non_tensor_batch.get("turn_counts")

        # Use multi-turn sampling if enabled and data is available
        if (
            self.multi_turn_enabled
            and all_turns_prompt_ids is not None
            and all_turns_reasoning_ids is not None
            and turn_counts is not None
        ):
            prompt_ids_list, reasoning_ids_list, group_ids = self._sample_multi_turn_pairs(
                all_turns_prompt_ids,
                all_turns_reasoning_ids,
                turn_counts,
                group_ids,
            )
            if len(prompt_ids_list) == 0:
                return {}
            # Multi-turn: each (prompt, reasoning) pair is unique, so use per-sample groups.
            group_ids = np.arange(len(prompt_ids_list), dtype=int)
        else:
            # Fallback to first_turn mode
            prompt_ids_list = batch.non_tensor_batch.get("first_turn_prompt_ids")
            reasoning_ids_list = batch.non_tensor_batch.get("first_turn_reasoning_ids")
            if prompt_ids_list is None or reasoning_ids_list is None:
                raise ValueError("Collapse metrics require first_turn_prompt_ids and first_turn_reasoning_ids.")
            if isinstance(prompt_ids_list, np.ndarray):
                prompt_ids_list = list(prompt_ids_list)
            if isinstance(reasoning_ids_list, np.ndarray):
                reasoning_ids_list = list(reasoning_ids_list)

        # Build mapping from group_id to column index
        unique_groups = np.unique(group_ids)
        gid_to_col = {int(gid): j for j, gid in enumerate(unique_groups)}

        N_prompts = len(unique_groups)
        if N_prompts < 2:
            # Need at least 2 prompts to compute meaningful metrics
            return {}

        try:
            # Extract representative prompts for each group
            prompts = self._extract_representative_prompts(
                group_ids,
                gid_to_col,
                prompt_ids_list,
            )

            pad_token_id = self._get_pad_token_id(batch)
            reasoning_ids, reasoning_mask = self._build_padded_token_batch(
                reasoning_ids_list, pad_token_id, batch.batch["input_ids"].device
            )

            # Compute cross log probabilities
            cross_log_probs, cross_log_probs_sum = self._compute_cross_log_probs(
                batch,
                actor_compute_log_prob_fn,
                prompts,
                unique_groups,
                reasoning_ids,
                reasoning_mask,
            )

            # Convert group_ids to column indices
            device = cross_log_probs.device
            col_ids = torch.tensor([gid_to_col[int(g)] for g in group_ids], device=device)

            matched, marginal = self._compute_log_prob_stats(cross_log_probs, col_ids)
            matched_sum, marginal_sum = self._compute_log_prob_stats(
                cross_log_probs_sum, col_ids
            )

            metrics = {}
            metrics.update(self._compute_mi_estimate(matched, marginal, N_prompts))
            metrics["collapse/mi_seq_estimate"] = (matched_sum - marginal_sum).mean().item()
            metrics.update(self._compute_retrieval_accuracy(cross_log_probs, col_ids, N_prompts))
            metrics.update(self._compute_reasoning_entropy(matched, marginal, matched_sum, marginal_sum))
            return metrics

        except Exception as e:
            print(f"[CollapseDetector] Error computing collapse metrics: {e}")
            return {}

    def _apply_mask(self, batch: DataProto, mask: np.ndarray) -> DataProto:
        """Apply a boolean mask to filter the batch."""
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        tensor_data = None
        if batch.batch is not None:
            tensor_data = batch.batch[mask_tensor]

        non_tensor_data = {}
        if batch.non_tensor_batch is not None:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    non_tensor_data[key] = val[mask]
                elif isinstance(val, list):
                    non_tensor_data[key] = [v for v, m in zip(val, mask) if m]
                else:
                    non_tensor_data[key] = val

        return DataProto(
            batch=tensor_data,
            non_tensor_batch=non_tensor_data if non_tensor_data else None,
            meta_info=batch.meta_info.copy() if batch.meta_info else {},
        )

    def _get_pad_token_id(self, batch: DataProto) -> int:
        attention_mask = batch.batch["attention_mask"]
        pad_positions = attention_mask == 0
        if pad_positions.any():
            first_pad_idx = pad_positions.nonzero()[0]
            return batch.batch["input_ids"][first_pad_idx[0], first_pad_idx[1]].item()
        return 0

    def _build_padded_token_batch(
        self,
        token_lists: List,
        pad_token_id: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = []
        for tokens in token_lists:
            if torch.is_tensor(tokens):
                lengths.append(tokens.numel())
            else:
                if isinstance(tokens, np.ndarray) and tokens.dtype == object:
                    tokens = tokens.tolist()
                lengths.append(len(tokens))
        max_len = max(max(lengths, default=0), 1)

        batch_size = len(token_lists)
        responses = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        response_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.float32, device=device
        )

        for i, tokens in enumerate(token_lists):
            if torch.is_tensor(tokens):
                ids = tokens.to(device=device, dtype=torch.long)
            else:
                if isinstance(tokens, np.ndarray) and tokens.dtype == object:
                    tokens = tokens.tolist()
                ids = torch.tensor(tokens, device=device, dtype=torch.long)
            if ids.numel() == 0:
                continue
            responses[i, : ids.numel()] = ids
            response_mask[i, : ids.numel()] = 1.0

        return responses, response_mask

    def _extract_representative_prompts(
        self,
        group_ids: np.ndarray,
        gid_to_col: Dict[int, int],
        prompt_ids_list: List,
    ) -> Dict[int, torch.Tensor]:
        """
        For each unique group, extract the prompt tokens from the first sample in that group.

        Returns:
            Dictionary mapping group_id to prompt token tensor
        """
        def _to_id_list(value: object) -> List[int]:
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            return list(value)

        prompts = {}

        for gid in gid_to_col.keys():
            # Find first sample with this group_id
            indices = np.where(group_ids == gid)[0]
            if indices.size == 0:
                raise ValueError(f"No samples found for group_id {gid}.")
            idx = int(indices[0])
            prompt_ids = _to_id_list(prompt_ids_list[idx])
            for other_idx in indices[1:]:
                other_prompt_ids = _to_id_list(prompt_ids_list[int(other_idx)])
                if prompt_ids != other_prompt_ids:
                    raise ValueError(
                        f"Prompt mismatch within group_id {gid}: index {idx} vs {int(other_idx)}."
                    )
            prompts[gid] = torch.tensor(prompt_ids, dtype=torch.long)

        return prompts

    def _compute_cross_log_probs(
        self,
        batch: DataProto,
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        prompts: Dict[int, torch.Tensor],
        unique_groups: np.ndarray,
        reasoning_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross log probabilities: ℓ_j(r_{i,k}) for all (i,k,j) pairs.

        For each reasoning r_{i,k} and each prompt x_j:
        1. Construct sequence [x_j | r_{i,k}]
        2. Compute teacher-forcing log prob
        3. Sum over reasoning tokens → ℓ_j(r_{i,k})

        Returns:
            cross_log_probs: (NK, N) per-token mean log prob tensor
            cross_log_probs_sum: (NK, N) per-sequence sum log prob tensor
        """
        NK = reasoning_ids.shape[0]
        N = len(unique_groups)
        device = reasoning_ids.device

        # Get reasoning tokens
        reasoning_len = reasoning_ids.shape[1]

        cross_log_probs = torch.zeros(NK, N, device=device)
        cross_log_probs_sum = torch.zeros(NK, N, device=device)

        for j, gid in enumerate(unique_groups):
            gid = int(gid)
            prompt = prompts[gid].to(device)  # (prompt_len,)
            prompt_len = prompt.shape[0]

            # Create cross batch: [x_j | r_{i,k}] for all reasoning sequences
            cross_input_ids_list = []
            cross_attention_mask_list = []
            cross_position_ids_list = []

            for start in range(0, NK, self.micro_batch_size):
                end = min(start + self.micro_batch_size, NK)
                batch_reasoning = reasoning_ids[start:end]  # (micro_bs, reasoning_len)
                micro_bs = batch_reasoning.shape[0]

                # Concatenate: [prompt | reasoning]
                prompt_expanded = prompt.unsqueeze(0).expand(micro_bs, -1)  # (micro_bs, prompt_len)
                cross_ids = torch.cat([prompt_expanded, batch_reasoning], dim=1)  # (micro_bs, prompt_len + reasoning_len)
                seq_len = cross_ids.shape[1]
                cross_attention = torch.ones_like(cross_ids)
                cross_attention[:, prompt_len:] = reasoning_mask[start:end].to(cross_attention.dtype)

                # Compute position_ids
                cross_position = torch.arange(seq_len, device=device).unsqueeze(0).expand(micro_bs, -1)

                cross_input_ids_list.append(cross_ids)
                cross_attention_mask_list.append(cross_attention)
                cross_position_ids_list.append(cross_position)

            # Concatenate all micro batches
            cross_input_ids = torch.cat(cross_input_ids_list, dim=0)  # (NK, seq_len)
            cross_attention_mask = torch.cat(cross_attention_mask_list, dim=0)
            cross_position_ids = torch.cat(cross_position_ids_list, dim=0)

            # Create cross batch DataProto
            cross_batch_data = TensorDict({
                "input_ids": cross_input_ids,
                "attention_mask": cross_attention_mask,
                "position_ids": cross_position_ids,
                "responses": reasoning_ids,
            }, batch_size=[NK])

            cross_batch = DataProto(
                batch=cross_batch_data,
                non_tensor_batch=batch.non_tensor_batch,
                meta_info=batch.meta_info.copy() if batch.meta_info else {},
            )

            # Compute log probabilities
            with torch.no_grad():
                output = compute_log_prob_fn(cross_batch)
                log_probs = output.batch["old_log_probs"]  # (NK, response_len)

            # Get response mask to sum over valid response tokens
            mask = reasoning_mask

            # Normalize by reasoning length to reduce length bias
            token_counts = mask.sum(dim=-1).clamp(min=1)
            seq_log_probs_sum = (log_probs * mask).sum(dim=-1)  # (NK,)
            seq_log_probs = seq_log_probs_sum / token_counts  # (NK,)
            cross_log_probs[:, j] = seq_log_probs
            cross_log_probs_sum[:, j] = seq_log_probs_sum

        return cross_log_probs, cross_log_probs_sum

    def _compute_log_prob_stats(
        self,
        cross_log_probs: torch.Tensor,
        col_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matched and marginal log-probabilities.
        """
        NK, N = cross_log_probs.shape
        device = cross_log_probs.device

        matched = cross_log_probs[torch.arange(NK, device=device), col_ids]
        marginal = torch.logsumexp(cross_log_probs, dim=1) - math.log(N)
        return matched, marginal

    def _compute_mi_estimate(
        self,
        matched: torch.Tensor,
        marginal: torch.Tensor,
        N_prompts: int,
    ) -> Dict[str, float]:
        """
        Compute mutual information estimate.

        Î(X;R) = E[log p(r|x) - log p_mix(r)]

        Args:
            matched: log p(r|x) for matched prompt
            marginal: log p_mix(r) under uniform prompt mixture
            N_prompts: Number of unique prompts
        Returns:
            Dictionary of MI-related metrics
        """
        matched_mean = matched.mean().item()
        marginal_mean = marginal.mean().item()
        mi = matched_mean - marginal_mean

        return {
            "collapse/mi_estimate": mi,
            "collapse/mi_upper_bound": math.log(N_prompts),  # Theoretical max
            "collapse/matched_log_prob_mean": matched_mean,
            "collapse/marginal_log_prob_mean": marginal_mean,
        }

    def _compute_retrieval_accuracy(
        self,
        cross_log_probs: torch.Tensor,
        col_ids: torch.Tensor,
        N_prompts: int,
    ) -> Dict[str, float]:
        """
        Compute retrieval accuracy.

        Acc = fraction where argmax_j ℓ_j(r) == true prompt column

        Args:
            cross_log_probs: (NK, N) tensor
            col_ids: Column indices for each sample's true prompt
            N_prompts: Number of unique prompts

        Returns:
            Dictionary of retrieval metrics
        """
        # Predict: which column (prompt) gives highest log prob
        predicted_cols = torch.argmax(cross_log_probs, dim=1)

        # Compare with true column indices
        correct = (predicted_cols == col_ids).float()
        accuracy = correct.mean().item()

        # Chance level depends on number of prompts
        chance_level = 1.0 / N_prompts

        return {
            "collapse/retrieval_accuracy": accuracy,
            "collapse/retrieval_chance_level": chance_level,
            "collapse/retrieval_above_chance": accuracy - chance_level,
        }
    
    def _compute_reasoning_entropy(
        self,
        matched: torch.Tensor,
        marginal: torch.Tensor,
        matched_sum: torch.Tensor,
        marginal_sum: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Estimate H(R|X) using the matched log-probabilities of the sampled reasoning.
        And Estimate reasoning entropy H(R) = H(R|X) + I(X;R).
        Per-token estimates are length-normalized; *_seq_est uses summed log probs.
        """
        matched_mean = matched.mean().item()
        marginal_mean = marginal.mean().item()
        conditional_entropy = -matched_mean
        mi_estimate = matched_mean - marginal_mean
        reasoning_entropy = conditional_entropy + mi_estimate
        matched_sum_mean = matched_sum.mean().item()
        marginal_sum_mean = marginal_sum.mean().item()
        conditional_entropy_seq = -matched_sum_mean
        reasoning_entropy_seq = -marginal_sum_mean
        return {
            "collapse/conditional_entropy_est": conditional_entropy,
            "collapse/reasoning_entropy_est": reasoning_entropy,
            "collapse/conditional_entropy_seq_est": conditional_entropy_seq,
            "collapse/reasoning_entropy_seq_est": reasoning_entropy_seq,
        }

    def _sample_multi_turn_pairs(
        self,
        all_turns_prompt_ids: np.ndarray,
        all_turns_reasoning_ids: np.ndarray,
        turn_counts: np.ndarray,
        group_ids: np.ndarray,
    ) -> Tuple[List, List, np.ndarray]:
        """
        Sample (x, r) pairs from multi-turn data based on the configured strategy.

        Args:
            all_turns_prompt_ids: shape (M,), each element is a list of prompt_ids per turn
            all_turns_reasoning_ids: shape (M,), each element is a list of reasoning_ids per turn
            turn_counts: shape (M,), number of valid turns per trajectory
            group_ids: shape (M,), group_id for each trajectory

        Returns:
            (sampled_prompt_ids, sampled_reasoning_ids, sampled_group_ids)
        """
        if self.multi_turn_strategy == "turn_uniform":
            return self._sample_turn_uniform(
                all_turns_prompt_ids, all_turns_reasoning_ids, turn_counts, group_ids
            )
        elif self.multi_turn_strategy == "trajectory_uniform":
            return self._sample_trajectory_uniform(
                all_turns_prompt_ids, all_turns_reasoning_ids, turn_counts, group_ids
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.multi_turn_strategy}")

    def _sample_turn_uniform(
        self,
        all_turns_prompt_ids: np.ndarray,
        all_turns_reasoning_ids: np.ndarray,
        turn_counts: np.ndarray,
        group_ids: np.ndarray,
    ) -> Tuple[List, List, np.ndarray]:
        """
        Turn-uniform sampling: uniform over all (trajectory, turn) pairs.
        Pr(m, t) = 1 / Σ_m T_m
        Longer trajectories contribute more samples.
        """
        # Expand all (trajectory_idx, turn_idx) pairs
        all_pairs = []
        for m, T_m in enumerate(turn_counts):
            for t in range(T_m):
                all_pairs.append((m, t))

        total_pairs = len(all_pairs)
        if total_pairs == 0:
            return [], [], np.array([], dtype=int)

        # Sample if num_samples is specified and smaller than total
        if self.num_samples is not None and self.num_samples < total_pairs:
            indices = np.random.choice(total_pairs, self.num_samples, replace=False)
            selected_pairs = [all_pairs[i] for i in indices]
        else:
            selected_pairs = all_pairs

        return self._extract_pairs(
            selected_pairs, all_turns_prompt_ids, all_turns_reasoning_ids, group_ids
        )

    def _sample_trajectory_uniform(
        self,
        all_turns_prompt_ids: np.ndarray,
        all_turns_reasoning_ids: np.ndarray,
        turn_counts: np.ndarray,
        group_ids: np.ndarray,
    ) -> Tuple[List, List, np.ndarray]:
        """
        Trajectory-uniform sampling: first uniform over trajectories, then uniform over turns.
        Pr(m, t) = 1/M · 1/T_m
        Each trajectory has equal weight regardless of length.
        """
        M = len(turn_counts)
        if M == 0:
            return [], [], np.array([], dtype=int)

        # Determine number of samples
        total_pairs = sum(turn_counts)
        if self.num_samples is None:
            num_to_sample = total_pairs
        else:
            num_to_sample = min(self.num_samples, total_pairs)

        # Sample: first pick trajectory uniformly, then pick turn uniformly within
        selected_pairs = []
        for _ in range(num_to_sample):
            m = np.random.randint(M)
            if turn_counts[m] > 0:
                t = np.random.randint(turn_counts[m])
                selected_pairs.append((m, t))

        return self._extract_pairs(
            selected_pairs, all_turns_prompt_ids, all_turns_reasoning_ids, group_ids
        )

    def _extract_pairs(
        self,
        pairs: List[Tuple[int, int]],
        all_turns_prompt_ids: np.ndarray,
        all_turns_reasoning_ids: np.ndarray,
        group_ids: np.ndarray,
    ) -> Tuple[List, List, np.ndarray]:
        """
        Extract prompt_ids and reasoning_ids for the selected (trajectory, turn) pairs.
        """
        sampled_prompt_ids = []
        sampled_reasoning_ids = []
        sampled_group_ids = []

        for m, t in pairs:
            prompt_ids = all_turns_prompt_ids[m][t]
            reasoning_ids = all_turns_reasoning_ids[m][t]
            sampled_prompt_ids.append(prompt_ids)
            sampled_reasoning_ids.append(reasoning_ids)
            sampled_group_ids.append(group_ids[m])

        return sampled_prompt_ids, sampled_reasoning_ids, np.array(sampled_group_ids, dtype=int)
