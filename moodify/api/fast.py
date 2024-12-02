import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..model import get_transformers_lyrics

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.state.model = get_transformers_lyrics()

@app.get("/gen_lyrics")
def gen_lyrics(user_prompt='Generate a song about war, just 4 lines', theme='sad'):
    model, tokenizer = app.state.model
    user_prompt = 'Generate a song about ' + user_prompt + '. Dont return anything but the raw lyrics'
    print(f'User prompt is: {user_prompt} and user theme is : {theme}')

    messages = [
        {"role": "system", "content": f'You are Qwen, created by Alibaba Cloud. You only generate song lyrics that are {theme}. '},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize= False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


@app.get("/")
def root():
    return 'Welcome to Moodify'
