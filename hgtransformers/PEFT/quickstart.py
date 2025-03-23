# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : quickstart.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/03/22 20:25
# @Description: 

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, lora_alpha=32, lora_dropout=.1
)

model = AutoModelForSeq2SeqLM.from_pretrained('bigscience/mt0-large')
model = get_peft_model(model=model, peft_config=peft_config)
model.print_trainable_parameters()