from datetime import datetime
import logging
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import torch

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

# model_path = "Helsinki-NLP/opus-mt-en-sk"
model_name = "saved"
scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)
model_path = os.path.join(scriptdir, model_name)

output_layer = 'loss:0'
input_node = 'Placeholder:0'

tokenizer = None
model = None


def _initialize():
    nltk.download('punkt')

    

    global tokenizer
    global model
    if tokenizer is None or model is None:
        
        _log_msg("Initializing model and tokenizer.")
        _log_msg("Torch version:" + str(torch.__version__))
        # local_files_only=True
        tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True) #, force_download=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True)#, force_download=True)

        # model.save_pretrained("./translateEnSk/saved/")
        # tokenizer.save_pretrained("./translateEnSk/saved/")

        
        _log_msg("Dynamic quantization of model.")
        # dynamic quantization for faster CPU inference
        model.to('cpu')
        torch.backends.quantized.engine = 'qnnpack' # ARM
        # torch.backends.quantized.engine = 'fbgemm' # x86
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)

        _log_msg("Model ready!")


def _log_msg(msg, debug=False):
    message = "{}: {}".format(datetime.now(),msg)

    if debug:
       logging.debug(message) 
       return

    logging.info(message)

def preprocess(text):
    return nltk.sent_tokenize(text)


def translate(text: str):

    _initialize()

    sentences = preprocess(text=text)

    _log_msg("Text length:" + str(len(text)), True)

    tok = tokenizer(sentences, return_tensors="pt",padding=True)
    translated = model.generate(**tok)

    translated = " ".join([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    
    _log_msg("Translated length: " + str(len(translated)), True)

    response = {
                'created': datetime.utcnow().isoformat(),
                'translations': translated 
            }

    print(response)

    _log_msg("Results: " + str(response))
    return response


if __name__ == "__main__":

    translate("My name is Sarah and I live in London. It is a very nice city.")