from cgitb import text
import logging
import azure.functions as func
import json

# Import helper script
from .predict import translate

# example url: http://localhost:7071/api/summarizeEn/?text="My name is Sarah and I live in London. It is a very nice city."
def main(req: func.HttpRequest) -> func.HttpResponse:
    text = req.params.get('text')
    logging.info('Text received: ' + text)

    results = translate(text)

    headers = {
        "Content-type": "application/json;charset='UTF-8'",
        "Access-Control-Allow-Origin": "*"
    }

    return func.HttpResponse(json.dumps(results, ensure_ascii=False).encode('utf8'), headers = headers)
