from transformers import pipeline,AutoModel,AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
import torch
import gradio


pipe=pipeline("text-classification", model="olipol/smaug_part1", tokenizer="olipol/smaug_part1")
gradio.Interface.from_pipeline(pipe).launch()


