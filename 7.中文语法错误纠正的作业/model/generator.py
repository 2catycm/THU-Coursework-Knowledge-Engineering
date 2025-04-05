import re
import torch 
import einops

# 定义BeamSearch生成结果
class BeamHypotheses:
    """
    A class to store current top `num_beams` best beam search generation output
    """
    def __init__(self, num_beams: int=5, length_penalty: float=0.7):
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.beams = [] # a list of (sequence, score) pair
        self.worst_score = 1e9 # storing the current worst score, which is always np.min([i[1] for i in self.beams]) when self.beams is not empty

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        """
        Add a generated sequence and its score
        The score, by default is \sum_{i=1}^n log(p(x_i|<bos>, x_1, ..., x_{i-1})), where {<bos>, x_1, ..., x_n} is the generated sequence and p(x_i|<bos>, x_1, ..., x_{i-1}) is given by decoder
        
        This method first calculate the penalized score (with length_penalty hyper-parameter)
        then updates the current self.beams and self.worst_score

        Returns:
            None
        """
        # TODO
        raise NotImplementedError

    def is_done(self, best_sum_logprobs: float, cur_len: int)-> bool:
        """
        check whether generation should stop

        Args:
            best_sum_logprobs: the best score at current timestep
            cur_len: the sequence length with best score at current timestep
        Returns:
            is_done: bool
        """
        # TODO
        raise NotImplementedError

class BeamSearchGenerator:
    def __init__(self, model, reverse_vocab_dict, device):
        self.oov_index = 0
        self.bos_index = 1
        self.eos_index = 2
        self.pad_index = 3
        self.model = model
        self.max_length = 200
        self.num_beams = 8
        self.length_penalty = 0.7
        self.vocab = reverse_vocab_dict
        self.vocab_size = model.vocab_size
        self.device = device

    def generate(self, **kwargs):
        source_mask = kwargs["source_mask"]
        encoder_outputs = self.model.encode(**kwargs)
        generated_sequence = self.beam_search(encoder_outputs, source_mask)
        generated_string = [re.sub("<bos>|<eos>|<pad>", "", " ".join(list(map(self.vocab.__getitem__, item.tolist())))) for i, item in enumerate(generated_sequence.detach().cpu())]
        generated_string = [re.sub("\s+", " ", item.strip()) for item in generated_string]
        return generated_string

    def beam_search(self, encoder_output, source_mask) -> torch.Tensor:
        """
        perform beam search to get the generated sequence with the highest score
        Arguments:
            - encoder_output: a torch tensor of shape (batch_size, sequence_length, hidden_size) representing the encoder output
            - mask: a torch tensor of shape (batch_size, sequence_length) representing the input mask
        Returns:
            output_sequence:
        """
        # get necessary info
        batch_size, max_length, num_beams, vocab_size = encoder_output.shape[0], self.max_length, self.num_beams, self.vocab_size

        # repeat encoder_output and source_mask with `num_beams`
        # b, s, h -> b*n, s, h
        encoder_output = einops.repeat(encoder_output, "b s h -> (b n) s h", n=num_beams)
        source_mask = einops.repeat(source_mask, "b s -> (b n) s", n=num_beams)

        # initialize variable for storing the top-`num_beams` generated sequence and corresponding scores at current timestep
        sequence = torch.zeros(batch_size * num_beams, 1, dtype=torch.long, device=self.device)
        sequence[:, 0] = self.bos_index # bos_index at beginning
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9 # set to a small value
        beam_scores = beam_scores.view(-1)

        # a flag tensor indicating whether generation is done for current sentence
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # initialize beams for each sequence
        hypotheses = [BeamHypotheses(num_beams, self.length_penalty) for _ in range(batch_size)]
        hidden_states = None

        # generate token by token
        ############################# 
        # TODO
        #################################

        # update each hypothesis
        for i in range(batch_size):
            if finished[i]:
                continue
            for j in range(num_beams):
                batch_beam_id = i * num_beams + j
                hypotheses[i].add(sequence[batch_beam_id], beam_scores[batch_beam_id].item())

        length = sequence.new(batch_size)
        best_sequence = []
        for i, hypo in enumerate(hypotheses):
            sorted_hypos = sorted(hypo.beams, key=lambda x: x[0])
            best_hypo = sorted_hypos[-1][1]
            length[i] = len(best_hypo)
            best_sequence.append(best_hypo)

        output_sequence = sequence.new(batch_size, max_length).fill_(self.eos_index)
        for i, one_sequence in enumerate(best_sequence):
            output_sequence[i, :length[i]] = one_sequence
        return output_sequence