import re
import torch
import einops

from typing import List, Tuple, Dict
from elmoformanylangs import Embedder
from .model import GECModel

class BeamHypotheses:
    def __init__(self, num_beams: int = 5, length_penalty: float = 0.7):
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.beams: List[Tuple] = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        score = sum_logprobs / (len(hyp) ** self.length_penalty)
        
        if len(self.beams) < self.num_beams:
            self.beams.append((score, hyp))
            self.worst_score = min(score, self.worst_score)
        else:
            if score > self.worst_score:
                worst_idx = min(enumerate(self.beams), key=lambda x: x[1][0])[0]
                self.beams[worst_idx] = (score, hyp)
                self.worst_score = min(score for score, _ in self.beams)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        if len(self.beams) < self.num_beams:
            return False
        cur_score = best_sum_logprobs / (cur_len ** self.length_penalty)
        return self.worst_score >= cur_score


class BeamSearchGenerator:
    def __init__(self, model: GECModel, # model.py 定义的模型
                 reverse_vocab_dict:Dict[int, str], # elmo 词典
                 device:str):
        self.oov_index = 0  # out of vocabulary
        self.bos_index = 1  # begin of sentence
        self.eos_index = 2  # end of sentence
        self.pad_index = 3  # padding
        self.model = model
        self.max_length = 200
        self.num_beams = 8
        self.length_penalty = 0.7
        self.vocab = reverse_vocab_dict
        self.vocab_size = len(reverse_vocab_dict) # 也就是 L 的大小。
        self.device = device

    def generate(self, source_mask:torch.Tensor, # shape (batch_size, sequence_length) representing the input mask
                 **kwargs):
        encoder_outputs = self.model.encode(**kwargs)
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

    def beam_search(self, encoder_output: torch.Tensor, # shape (batch_size, sequence_length, hidden_size) representing the encoder output
                    source_mask: torch.Tensor # shape (batch_size, sequence_length) representing the input mask
        ) -> torch.Tensor: # int ids, shape (batch_size, max_length)
        """
        perform beam search to get the generated sequence with the highest score
        """
        # 0. get necessary info
        batch_size, seq_length, hidden_size = encoder_output.shape
        num_beams = self.num_beams
        vocab_size = self.vocab_size
        max_length = self.max_length

        # 1. prepare the encoder output and source mask 
        encoder_output = einops.repeat(encoder_output, "b s h -> (b n) s h", n=num_beams)
        source_mask = einops.repeat(source_mask, "b s -> (b n) s", n=num_beams)

        # 2. 初始化为(batch*beam, 1)格式, storing the top-`num_beams` generated sequence
        sequence = torch.full(
            (batch_size * num_beams, 1),
            self.bos_index, # 填满这个 begin of sentence
            dtype=torch.long,
            device=self.device
        )
        # 3. beam scores, corresponding scores at current timestep
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # 4. a flag tensor indicating whether generation is done for current sentence
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        hypotheses = [BeamHypotheses(num_beams, self.length_penalty) for _ in range(batch_size)]
        hidden_states = None
        
        
        # generate token by token
        cur_len = 1
        while cur_len < max_length:
            # 直接使用sequence作为2D输入
            decoder_output = self.model.decode(
                encoder_output,
                source_mask,
                target_input_ids=sequence,  # (batch*beam, seq_len)
                target_mask=None,
                hidden_states=hidden_states
            )
            
            next_token_logits = decoder_output[:, -1, :]
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            
            next_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            next_tokens = torch.argsort(next_scores, dim=1, descending=True)[:, :num_beams]

            for batch_idx in range(batch_size):
                if finished[batch_idx].item():
                    continue
                    
                batch_next_scores = next_scores[batch_idx]
                batch_next_tokens = next_tokens[batch_idx]
                
                for beam_token_rank, (token_id, token_score) in enumerate(
                    zip(batch_next_tokens, batch_next_scores[batch_next_tokens])
                ):
                    beam_id = batch_idx * num_beams + beam_token_rank
                    
                    if token_id.item() == self.eos_index:
                        hypotheses[batch_idx].add(
                            sequence[beam_id].clone(),
                            beam_scores[beam_id].item()
                        )
                    else:
                        new_score = token_score
                        beam_scores[beam_id] = new_score
                        # 直接将新token赋值到当前时间步位置，使用 squeeze() 消除多余维度
                        extension = cur_len - sequence.size(1) + 1
                        if extension > 0:
                            pad = torch.zeros((sequence.size(0), extension), dtype=sequence.dtype, device=sequence.device)
                            sequence = torch.cat([sequence, pad], dim=1)
                        sequence[beam_id, cur_len] = token_id.item()
                
                finished[batch_idx] = hypotheses[batch_idx].is_done(
                    torch.max(next_scores[batch_idx]).item(),
                    cur_len
                )
            
            if torch.all(finished).item():
                break
                
            cur_len += 1
            
        # update each hypothesis
        for i in range(batch_size):
            if finished[i]:
                continue
            for j in range(num_beams):
                batch_beam_id = i * num_beams + j
                hypotheses[i].add(
                    sequence[batch_beam_id],
                    beam_scores[batch_beam_id].item()
                )

        length = sequence.new(batch_size)
        best_sequence = []
        for i, hypo in enumerate(hypotheses):
            sorted_hypos = sorted(hypo.beams, key=lambda x: x[0])
            best_hypo = sorted_hypos[-1][1]
            length[i] = best_hypo.size(1)
            best_sequence.append(best_hypo)

        # 输出保持(batch_size, max_length)格式
        output_sequence = sequence.new(batch_size, max_length).fill_(self.eos_index)
        for i, one_sequence in enumerate(best_sequence):
            output_sequence[i, :length[i]] = one_sequence  # 直接使用(seq_len,)序列
        return output_sequence
