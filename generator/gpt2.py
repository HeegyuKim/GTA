from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2PreTrainedModel, LogitsProcessorList, LogitsProcessor, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class GPT2Generator:
    model_name: str
    num_return_sequences: int
    max_tokens: int
    p: float = 1.0
    device: str = "cpu"
    float16: bool = False

    def __post_init__(self):
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
        if self.float16:
            self.model = self.model.half()

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] + self.max_tokens >= 1024:
            print("too long context ", inputs["input_ids"].shape[1])
        outputs = self.model.generate(
            **inputs,
            num_return_sequences=self.num_return_sequences,
            max_new_tokens=self.max_tokens,
            top_p=self.p,
            do_sample=True,
            early_stopping=True,
            pad_token_id=50256
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(prompt, outputs)
        outputs = [x[len(prompt) - 1:] for x in outputs]
        return outputs