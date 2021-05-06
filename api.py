#!/usr/bin/env python
import os
import tensorflow as tf
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

from fastapi import FastAPI, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional, Any, List
from types import SimpleNamespace

import cipherTypeDetection.eval as cipherEval
import cipherTypeDetection.config as config
from cipherTypeDetection.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
from cipherTypeDetection.ensembleModel import EnsembleModel


# init fast api
app = FastAPI()
models = {}

# allow cors
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # todo: remove later


@app.on_event("startup")
async def startup_event():
    """The models are loaded with hardcoded names. Change in future if multiple models are available."""
    model_path = "data/models"
    models["Transformer"] = (tf.keras.models.load_model(os.path.join(model_path, "t96_transformer_final_100.h5"), custom_objects={
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'MultiHeadSelfAttention': MultiHeadSelfAttention,
        'TransformerBlock': TransformerBlock}), False, True)
    models["FFNN"] = (tf.keras.models.load_model(os.path.join(model_path, "t128_ffnn_final_100.h5")), True, False)
    models["LSTM"] = (tf.keras.models.load_model(os.path.join(model_path, "t129_lstm_final_100.h5")), False, True)
    optimizer = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon,
                     amsgrad=config.amsgrad)
    for _, item in models.items():
        item[0].compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[
            "accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
    # TODO add this in production when having at least 32 GB RAM
    # with open(os.path.join(model_path, "t99_rf_final_100.h5"), "rb") as f:
    #     models["RF"] = (pickle.load(f), True, False)
    with open(os.path.join(model_path, "t128_nb_final_100.h5"), "rb") as f:
        models["NB"] = (pickle.load(f), True, False)


class APIResponse(BaseModel):
    """Define api response model."""
    success: bool = True
    payload: Optional[Any] = {}
    error_msg: Optional[str] = None


@app.exception_handler(Exception)
async def exception_handler(request, exc):
    """Define exception response format."""
    return JSONResponse({"success": False, "payload": None, "error_msg": str(exc)}, status_code=status.HTTP_400_BAD_REQUEST)
# TODO: does not work (exceptions are still thrown), specific exceptions work -- todo: FIX ;D

# todo: custom error handling for unified error responses and no closing on error
# https://fastapi.tiangolo.com/tutorial/handling-errors/#override-the-default-exception-handlers


@app.get("/get_available_architectures", response_model=APIResponse)
async def get_available_architectures():
    return {"success": True, "payload": models.keys()}


@app.get("/evaluate/single_line/ciphertext", response_model=APIResponse)
async def evaluate_single_line_ciphertext(ciphertext: str, architecture: List[str] = Query(None)):
    if not 0 < len(architecture) < 5:
        return JSONResponse({"success": False, "payload": None, "error_msg": "The number of architectures must be between 1 and 5."},
                            status_code=status.HTTP_400_BAD_REQUEST)
    cipher_types = get_cipher_types_to_use(["aca"])  # aca stands for all implemented ciphers
    if len(architecture) == 1:
        if architecture[0] not in models.keys():
            return JSONResponse({"success": False, "payload": None, "error_msg": "The architecture '%s' does not exist!" % architecture[0]},
                                status_code=status.HTTP_400_BAD_REQUEST)
        model, feature_engineering, pad_input = models[architecture[0]]
        cipherEval.architecture = architecture[0]
        config.FEATURE_ENGINEERING = feature_engineering
        config.PAD_INPUT = pad_input
    else:
        cipher_indices = []
        for cipher_type in cipher_types:
            cipher_indices.append(config.CIPHER_TYPES.index(cipher_type))
        model_list = []
        architecture_list = []
        for val in architecture:
            if val not in models.keys():
                return JSONResponse({"success": False, "payload": None, "error_msg": "The architecture '%s' does not exist!" % val},
                                    status_code=status.HTTP_400_BAD_REQUEST)
            if len(set(models.keys())) != len(models.keys()):
                return JSONResponse({"success": False, "payload": None, "error_msg": "Multiple architectures of the same type are not "
                                                                                     "allowed!"}, status_code=status.HTTP_400_BAD_REQUEST)
            model_list.append(models[val][0])
            architecture_list.append(val)
        cipherEval.architecture = "Ensemble"
        model = EnsembleModel(model_list, architecture_list, "weighted", cipher_indices)

    try:
        # call prediction method
        result = cipherEval.predict_single_line(SimpleNamespace(
            ciphertext = ciphertext,
            file = None, # todo: needs fileupload first (either set ciphertext OR file, never both)
            ciphers = cipher_types,
            batch_size = 128,
            verbose = False
        ), model)
    except BaseException as e:
        # only use these lines for debugging. Never in production environment due to security reasons!
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "payload": None, "error_msg": repr(e)}, status_code=500)
        # return JSONResponse(None, status_code=500)
    return {"success": True, "payload": result}


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
