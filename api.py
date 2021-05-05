#!/usr/bin/env python

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional, Any

from io import StringIO
from contextlib import redirect_stdout
from types import SimpleNamespace

import cipherTypeDetection.eval as cipherEval
import cipherTypeDetection.config as config


# init fast api
app = FastAPI()

# allow cors
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) # todo: remove later


# init ncid models
# to use for Ensemble
# todo ^^


# define api response model
class APIResponse(BaseModel):
    success: bool = True
    payload: Optional[Any] = {}
    error_msg: Optional[str] = None

# define exception response format
@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return JSONResponse({"success": False, "payload": None, "error_msg": str(exc)}, status_code=400)
# TODO: does not work (exceptions are still thrown), specific exceptions work -- todo: FIX ;D

# todo: custom error handling for unified error responses and no closing on error
# https://fastapi.tiangolo.com/tutorial/handling-errors/#override-the-default-exception-handlers


@app.get("/get_available_architectures", response_model=APIResponse)
async def get_available_architectures():
    return { "success": True, "payload": architecture_models }

@app.get("/evaluate/single_line/ciphertext", response_model=APIResponse)
async def evaluate_single_line_ciphertext(ciphertext: str, architecture: str):

    # todo?: if architecture == 'Ensemble': (eval.py line 564)

    # catch std output from function cuz
    # it's just cli printed (not returned)

    try:
        f = StringIO()
        with redirect_stdout(f):

            # call prediction method
            result = cipherEval.predict_single_line(SimpleNamespace(
                ciphertext = ciphertext,
                file = None, # todo: needs fileupload first (either set ciphertext OR file, never both)
                ciphers = get_cipher_types_to_use(["aca"]), # aca stands for all implemented ciphers
                batch_size = 128,
                verbose = True
            ), load_model(architecture))

        out = f.getvalue()
    except BaseException as e:
        return build_answer(False, None, repr(e))

    return { "success": True, "payload": out } # result }


###########################################################

def get_cipher_types_to_use(cipher_types):
    if config.MTC3 in cipher_types:
        del cipher_types[cipher_types.index(config.MTC3)]
        cipher_types.append(config.CIPHER_TYPES[0])
        cipher_types.append(config.CIPHER_TYPES[1])
        cipher_types.append(config.CIPHER_TYPES[2])
        cipher_types.append(config.CIPHER_TYPES[3])
        cipher_types.append(config.CIPHER_TYPES[4])
    if config.ACA in cipher_types:
        del cipher_types[cipher_types.index(config.ACA)]
        cipher_types.append(config.CIPHER_TYPES[0])
        cipher_types.append(config.CIPHER_TYPES[1])
        cipher_types.append(config.CIPHER_TYPES[2])
        cipher_types.append(config.CIPHER_TYPES[3])
        cipher_types.append(config.CIPHER_TYPES[4])
        cipher_types.append(config.CIPHER_TYPES[5])
        cipher_types.append(config.CIPHER_TYPES[6])
        cipher_types.append(config.CIPHER_TYPES[7])
        cipher_types.append(config.CIPHER_TYPES[8])
        cipher_types.append(config.CIPHER_TYPES[9])
        cipher_types.append(config.CIPHER_TYPES[10])
        cipher_types.append(config.CIPHER_TYPES[11])
        cipher_types.append(config.CIPHER_TYPES[12])
        cipher_types.append(config.CIPHER_TYPES[13])
        cipher_types.append(config.CIPHER_TYPES[14])
        cipher_types.append(config.CIPHER_TYPES[15])
        cipher_types.append(config.CIPHER_TYPES[16])
        cipher_types.append(config.CIPHER_TYPES[17])
        cipher_types.append(config.CIPHER_TYPES[18])
        cipher_types.append(config.CIPHER_TYPES[19])
        cipher_types.append(config.CIPHER_TYPES[20])
        cipher_types.append(config.CIPHER_TYPES[21])
        cipher_types.append(config.CIPHER_TYPES[22])
        cipher_types.append(config.CIPHER_TYPES[23])
        cipher_types.append(config.CIPHER_TYPES[24])
        cipher_types.append(config.CIPHER_TYPES[25])
        cipher_types.append(config.CIPHER_TYPES[26])
        cipher_types.append(config.CIPHER_TYPES[27])
        cipher_types.append(config.CIPHER_TYPES[28])
        cipher_types.append(config.CIPHER_TYPES[29])
        cipher_types.append(config.CIPHER_TYPES[30])
        cipher_types.append(config.CIPHER_TYPES[31])
        cipher_types.append(config.CIPHER_TYPES[32])
        cipher_types.append(config.CIPHER_TYPES[33])
        cipher_types.append(config.CIPHER_TYPES[34])
        cipher_types.append(config.CIPHER_TYPES[35])
        cipher_types.append(config.CIPHER_TYPES[36])
        cipher_types.append(config.CIPHER_TYPES[37])
        cipher_types.append(config.CIPHER_TYPES[38])
        cipher_types.append(config.CIPHER_TYPES[39])
        cipher_types.append(config.CIPHER_TYPES[40])
        cipher_types.append(config.CIPHER_TYPES[41])
        cipher_types.append(config.CIPHER_TYPES[42])
        cipher_types.append(config.CIPHER_TYPES[43])
        cipher_types.append(config.CIPHER_TYPES[44])
        cipher_types.append(config.CIPHER_TYPES[45])
        cipher_types.append(config.CIPHER_TYPES[46])
        cipher_types.append(config.CIPHER_TYPES[47])
        cipher_types.append(config.CIPHER_TYPES[48])
        cipher_types.append(config.CIPHER_TYPES[49])
        cipher_types.append(config.CIPHER_TYPES[50])
        cipher_types.append(config.CIPHER_TYPES[51])
        cipher_types.append(config.CIPHER_TYPES[52])
        cipher_types.append(config.CIPHER_TYPES[53])
        cipher_types.append(config.CIPHER_TYPES[54])
        cipher_types.append(config.CIPHER_TYPES[55])
    return cipher_types

architecture_models = {
    "FFNN": "data/models/t128_ffnn_final_100.h5",
    # "CNN": "", # model not available yet
    "LSTM": "data/models/t129_lstm_final_100.h5",
    # "DT": "", # model not available yet
    "NB": "data/models/t128_nb_final_100.h5",
    "RF": "data/models/t99_rf_final_100.h5",
    # "ET": "", # model not available yet
    "Transformer": "data/models/t96_transformer_final_100.h5",
    # "Ensemble": # model not available yet
}

def load_model(architecture):
    cipherEval.architecture = architecture
    cipherEval.args = SimpleNamespace()

    for arch, model_file in architecture_models.items():
        if arch == architecture:
            cipherEval.args.model = model_file

    # if cipherEval.args.model is None:
        # todo: raise error (architecture not available)

    return cipherEval.load_model()
