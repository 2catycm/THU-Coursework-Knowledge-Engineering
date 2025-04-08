import re
import torch
import einops

from typing import List, Tuple, Dict
from elmoformanylangs import Embedder
from .model import GECModel


class BeamHypotheses:
    """
    A class to store current top `num_beams` best beam search generation output
    """

    def __init__(self, num_beams: int = 5, length_penalty: float = 0.7):
        self.length_penalty = length_penalty
        self.num_beams = num_beams  # a list of (sequence, score) pair
        self.beams: List[Tuple] = []
        self.worst_score = 1e9  #  # storing the current worst score,
        # which is always np.min([i[1] for i in self.beams]) when self.beams is not empty

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        """
        Add a generated sequence and its score
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty  # apply length penalty
        if len(self.beams) < self.num_beams:
            self.beams.append((hyp, score))
            if score < self.worst_score:
                self.worst_score = score
        else:
            if score > self.worst_score:
                self.beams.sort(key=lambda x: x[1])  # Sort by score
                self.beams = self.beams[:-1]  # Remove the worst
                self.beams.append(
                    (hyp, score)
                )  # 其实应该搞一个堆数据结构，但是没关系，排序肯定对。
                self.worst_score = min([x[1] for x in self.beams])

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        Check whether generation should stop
        """
        if len(self.beams) < self.num_beams:
            return False
        cur_score = best_sum_logprobs / (cur_len**self.length_penalty)
        return self.worst_score >= cur_score


class BeamSearchGenerator:
    def __init__(
        self,
        model: GECModel,  # model.py 定义的模型
        reverse_vocab_dict: Dict[int, str],  # elmo 词典
        device: str,
    ):
        self.oov_index = 0  # out of vocabulary
        self.bos_index = 1  # begin of sentence
        self.eos_index = 2  # end of sentence
        self.pad_index = 3  # padding
        self.model = model
        self.max_length = 200
        self.num_beams = 8
        self.length_penalty = 0.7
        self.vocab = reverse_vocab_dict
        self.vocab_size = len(reverse_vocab_dict)  # 也就是 L 的大小。
        self.device = device

    def generate(
        self,
        source_inputs: List[List[str]],  # a list of input text
        source_mask: torch.Tensor,  # shape (batch_size, sequence_length) representing the input mask
        **kwargs,
    ):
        encoder_outputs = self.model.encode(
            source_inputs=source_inputs, source_mask=source_mask, **kwargs
        )
        generated_sequence = self.beam_search(encoder_outputs, source_mask)
        generated_string = [
            re.sub(
                "<bos>|<eos>|<pad>",
                "",
                " ".join(list(map(self.vocab.__getitem__, item.tolist()))),
            )
            for i, item in enumerate(generated_sequence.detach().cpu())
        ]
        generated_string = [
            re.sub(r"\s+", " ", item.strip()) for item in generated_string
        ]
        return generated_string

    def beam_search(
        self,
        encoder_output: torch.Tensor,  # shape (batch_size, sequence_length, hidden_size) representing the encoder output
        source_mask: torch.Tensor,  # shape (batch_size, sequence_length) representing the input mask
    ) -> torch.Tensor:  # int ids, shape (batch_size, max_length)
        """
        perform beam search to get the generated sequence with the highest score
        """
        # 0. get necessary info
        batch_size, seq_length, hidden_size = encoder_output.shape
        num_beams = self.num_beams
        vocab_size = self.vocab_size
        max_length = self.max_length

        # 1. prepare the encoder output and source mask
        encoder_output = einops.repeat(
            encoder_output, "b s h -> (b n) s h", n=num_beams
        )
        source_mask = einops.repeat(source_mask, "b s -> (b n) s", n=num_beams)

        # 2. 初始化为(batch*beam, 1)格式, storing the top-`num_beams` generated sequence
        sequence = torch.full(
            (batch_size * num_beams, 1),
            self.bos_index,  # 填满这个 begin of sentence
            dtype=torch.long,
            device=self.device,
        )
        # 3. beam scores, corresponding scores at current timestep
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # 4. a flag tensor indicating whether generation is done for current sentence
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        hypotheses = [
            BeamHypotheses(num_beams, self.length_penalty) for _ in range(batch_size)
        ]
        hidden_states = None  # 有点奇怪

        # Beam search loop
        for cur_len in range(1, max_length):
            # Get decoder output
            decoder_output, new_hidden_states = self.model.decode(
                encoder_outputs=encoder_output,
                source_mask=source_mask,
                target_input_ids=sequence,
                target_mask=torch.ones_like(sequence).to(self.device),
                hidden_states=hidden_states
            )

            # Get logits for the next token
            logits = decoder_output[:, -1, :]  # (batch_size * num_beams, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Add log_probs to beam_scores
            next_scores = beam_scores.unsqueeze(-1) + log_probs  # (batch_size * num_beams, vocab_size)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # Get top k scores and indices
            next_scores, next_indices = torch.topk(next_scores, num_beams, dim=1)
            next_beam_indices = next_indices // vocab_size
            next_token_indices = next_indices % vocab_size

            # Update beam_scores
            beam_scores = next_scores.view(-1)

            # Update sequence
            sequence = sequence[next_beam_indices.view(-1)]
            sequence = torch.cat(
                [sequence, next_token_indices.view(-1, 1)], dim=1
            )

            # Update hidden states
            if new_hidden_states is not None:
                hidden_states = new_hidden_states[:, next_beam_indices.view(-1), :]
                # hidden_states = new_hidden_states

            # Check for EOS tokens
            eos_in_beam = (next_token_indices == self.eos_index).any(dim=1)
            for i in range(batch_size):
                if eos_in_beam[i].any():
                    beam_idx = i * num_beams
                    for j in range(num_beams):
                        if beam_idx + j < next_token_indices.size(0) and next_token_indices[beam_idx + j] == self.eos_index:
                            hypotheses[i].add(
                                sequence[beam_idx + j].clone(),
                                beam_scores[beam_idx + j].item()
                            )
                    if len(hypotheses[i]) >= num_beams:
                        finished[i] = True

            if finished.all():
                break

        # Get the best sequence from hypotheses
        best_sequences = []
        for i in range(batch_size):
            if len(hypotheses[i]) > 0:
                best_sequences.append(hypotheses[i].beams[-1][0])
            else:
                best_sequences.append(sequence[i * num_beams])

        # Pad sequences to max_length
        output_sequence = torch.full(
            (batch_size, max_length), self.eos_index, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(best_sequences):
            seq_len = min(len(seq), max_length)
            output_sequence[i, :seq_len] = seq[:seq_len]

        return output_sequence