# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import click
import pandas as pd
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import torch
from torch import FloatTensor, LongTensor
from tqdm import tqdm
import os
import jsonlines
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2PreTrainedModel, LogitsProcessorList, LogitsProcessor, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer


class GPT2LMHeadModelForGate(GPT2LMHeadModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.ctg_warper = None

        # print(config)

    def _get_logits_warper(self, *args, **kwargs) -> LogitsProcessorList:
        # print(kwargs)
        # print("top-k", self.config.top_k)
        # top-k가 기본값으로 50이 되버린다;;;
        # self.config.top_k = None
        warpers = super()._get_logits_warper(*args, **kwargs)
        if self.ctg_warper is not None:
            self.ctg_warper.original_warpers = warpers
            return self.ctg_warper
        else:
            return warpers

class GatedLogitsProcessor(LogitsProcessor):
    def __init__(self, generation_tokenizer, classifier_tokenizer, classifier, 
                 post_processor: LogitsProcessor, label_index: int = 1, 
                 gate_threshold: float = 0.5,
                 device: str = 'cpu') -> None:
        super().__init__()
        self.post_processor = post_processor
        self.gate_threshold = gate_threshold
        self.original_warpers = None
        self.generation_tokenizer = generation_tokenizer
        self.classifier_tokenizer = classifier_tokenizer
        self.classifier = classifier
        self.label_index = label_index
        self.device = device

    def _classify_text(self, texts):
        inputs = self.classifier_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.classifier(**inputs).logits.softmax(-1)[:, self.label_index]
        outputs = (outputs > self.gate_threshold).float()
        return outputs
    
    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        warped_scores = self.original_warpers(input_ids, scores)
        next_tokens = torch.multinomial(warped_scores.softmax(-1), num_samples=1) # (b, 1)
        current_ids = torch.cat([input_ids, next_tokens], 1) # (b, s + 1)
        current_texts = self.generation_tokenizer.batch_decode(current_ids, skip_special_tokens=True)
        gate_output = self._classify_text(current_texts)
        

        logits = torch.ones_like(scores, device=scores.device) * -50000
        logits[torch.range(0, logits.shape[0] - 1, device=scores.device, dtype=torch.long), next_tokens.long().squeeze(1)] = 0

        if gate_output.sum().item() > 0:
            print(current_texts)
            print("toxic!", gate_output.cpu())
            gate_output = gate_output.unsqueeze(1)

            guided = self.post_processor(input_ids, scores)
            gated_scores = logits * (1 - gate_output) + gate_output * guided
            return self.original_warpers(input_ids, gated_scores)
        else:
            return logits
        
class DExpertLogitsProcessor(LogitsProcessor):

    def __init__(self, alpha, expert, anti_expert) -> None:
        super().__init__()
        self.alpha = alpha
        self.expert = expert
        self.anti_expert = anti_expert

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor: # of shape (batch_size, config.vocab_size)
        expert_logits = self.expert(input_ids).logits[:, -1]
        anti_expert_logits = self.anti_expert(input_ids).logits[:, -1]
        return scores + self.alpha * (expert_logits - anti_expert_logits)


@dataclass
class DExpertGenerator:
    model_name: str
    expert_model_name: str
    anti_expert_model_name: str
    num_return_sequences: int
    max_tokens: int
    min_tokens: int = 5
    p: float = 1.0
    alpha: float = 1.0
    classifier_model_name: Optional[str] = None
    device: str = "cpu"
    float16: bool = False
    gate_threshold: float = 0.5

    def __post_init__(self):
        device = self.device
        self.model = GPT2LMHeadModelForGate.from_pretrained(self.model_name).to(device).eval()
        self.expert = GPT2LMHeadModel.from_pretrained(self.expert_model_name).to(device).eval()
        self.anti_expert = GPT2LMHeadModel.from_pretrained(self.anti_expert_model_name).to(device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        processor = DExpertLogitsProcessor(
            alpha=self.alpha,
            expert=self.expert,
            anti_expert=self.anti_expert
        )
        if self.classifier_model_name and self.classifier_model_name != "no":
            processor = GatedLogitsProcessor(
                self.tokenizer,
                AutoTokenizer.from_pretrained(self.classifier_model_name),
                AutoModelForSequenceClassification.from_pretrained(self.classifier_model_name).to(device).eval(),
                processor,
                gate_threshold=self.gate_threshold,
                device=device
            )

        # self.logits_processors = LogitsProcessorList([processor])
        self.model.ctg_warper = processor
    
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            num_return_sequences=self.num_return_sequences,
            max_new_tokens=self.max_tokens,
            min_length=self.min_tokens + inputs["input_ids"].shape[1],
            top_p=self.p,
            do_sample=True,
            early_stopping=True,
            pad_token_id=50256
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [x[len(prompt):] for x in outputs]
        return outputs