
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from .modeling_gpt2 import GPT2LMHeadModel
import torch


@dataclass
class GeDiGenerator:
    model_name: str
    gate_model_name: str
    num_return_sequences: int
    max_tokens: int
    min_tokens: int = 5
    batch_size: int = 1
    disc_weight: int = 30
    logits_scale: float = 10.0
    gedi_model_name: str = './models/GeDi/cc_lm_detox'
    filter_p: float = 0.8
    target_p: float = 0.8
    top_k: int = 50
    top_p: float = 1.0
    device: str = "cpu"
    float16: bool = False
    class_bias: float = 0.0
    gate_threshold: float = 0.5

    def __post_init__(self):
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.gedi_model = GPT2LMHeadModel.from_pretrained(self.gedi_model_name).eval().to(self.device)

        if self.gate_model_name is not None and self.gate_model_name != "no":
            self.gate_tokenizer = AutoTokenizer.from_pretrained(self.gate_model_name)
            self.gate_model = AutoModelForSequenceClassification.from_pretrained(self.gate_model_name).eval().to(self.device)
        else:
            self.gate_model, self.gate_tokenizer = None, None

        if self.float16:
            if self.gedi_model:
                self.gedi_model.half()
            if self.model:
                self.model.half()
            if self.gate_model:
                self.gate_model.half()
    
    def generate(self, prompt):
        code_desired = "dirty"
        code_undesired = "clean"
        text_ids = self.tokenizer.encode(prompt)
        encoded_prompts=torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
        
        attr_class = 1

        generated_sequence = self.model.generate(
            input_ids=encoded_prompts,
            pad_lens=None,
            max_length=min(1024, encoded_prompts.shape[1] + self.max_tokens),
            min_length=encoded_prompts.shape[1] + self.min_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            # repetition_penalty= 1.2,
            # rep_penalty_scale= 10,
            eos_token_ids = [50256],
            pad_token_id = 50256,
            do_sample= True,
            penalize_cond= True,
            gedi_model=self.gedi_model,
            tokenizer=self.tokenizer,
            disc_weight=self.disc_weight,
            filter_p=self.filter_p,
            target_p=self.target_p,
            class_bias=self.class_bias,
            attr_class=attr_class,
            code_0 = code_desired,
            code_1 = code_undesired,
            multi_code=None,
            num_return_sequences=self.batch_size,
            classifier_model=self.gate_model, 
            classifier_tokenizer=self.gate_tokenizer,
            gate_threshold=self.gate_threshold,
            )

        text = self.tokenizer.decode(generated_sequence.tolist()[0], skip_special_tokens=True)
        text = text[len(prompt) - 1:]
        # print(text)
        return [text]
    
