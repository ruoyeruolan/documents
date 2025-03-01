# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : load_peft_adapter.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/19 20:24
@Description: 
"""

import token
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, OPTForCausalLM
from peft.tuners.lora.config import LoraConfig
from peft.config import PeftConfig

def load_peft_adapter():
    
    peft_model_id = 'ybelkada/opt-350m-lora'
    model = AutoModelForCausalLM.from_pretrained(peft_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))  # may report error with quantization_config

    # Add new adapter & set adapter to be used
    model_id = "facebook/opt-350m"
    # peft_model_id = "ybelkada/opt-350m-lora"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    lora_config = LoraConfig(
        target_modules=['q_pro', 'k_proj'],
        init_lora_weights=False
    )
    model.add_adapter(lora_config, adapter_name='adapter_1')
    model.add_adapter(lora_config, adapter_name="adapter_2")


    model.set_adapter('adapter_1')


    # Enable and diaable adapters
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer('Hello', return_tensors='pt')

    model = AutoModelForCausalLM.from_pretrained(model_id)
    peft_config = PeftConfig.from_pretrained(peft_model_id)

    peft_config.init_lora_weights = False
    model.add_adapter(peft_config)
    model.enable_adapters()
    output = model.generate(**inputs)
    # tensor([[    2, 31414,   223,   223,   223,   223,   223,   223,   223,   223,
    #        223,   223,   223,   223,   223,   223,   223,   223,   223,   223,
    #        223,   223]])


    # disable adapter module
    model.disable_adapters()
    output = model.generate(**inputs)
    # tensor([[    2, 31414,     6,    38,   437,    10,    92, 12750,     7,    42,
    #       2849,     4,    38,   437,   546,    13,    10,   205,   317,     7,
    #        386,     4]])
