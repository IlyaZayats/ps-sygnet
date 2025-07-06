import http
import uuid

import keras
import numpy as np
from flask import Flask, Response
from flask import request
import base64
import io
import zlib
import os
import json

from predict.classification import ClassificationWeights
from predict.classification import get_prediction
from predict.classification import get_grad_cam

from predict.sygmentation import SygmentationWeights
from predict.sygmentation import get_heatmap

from psql_repo.psql_repo import login
from s3_repo.s3_repo import S3Adapter
from server.psql_repo.models import Research
from server.psql_repo.psql_repo import add_research, get_researches

app = Flask(__name__)

sygmentation_weights = SygmentationWeights()
classification_weights = ClassificationWeights()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

db_url = os.getenv('DB_URL')
engine = create_engine(db_url)
SessionLocal = sessionmaker(bind=engine)

db = SessionLocal()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = "us-east-1"
BUCKET_NAME = "research-sygnet"

s3 = S3Adapter(AWS_ACCESS_KEY, AWS_SECRET_KEY, region=AWS_REGION)


def compress_nparr(nparr):
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed

def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring))).reshape(192,192,128)


@app.route("/login", methods=['POST'])
def login_handler():
    print("Got Login request from: " + request.remote_addr)

    request_body = request.get_json()

    user_id, ok = login(db, request_body['login'], request_body['password'])
    if not ok:
        return {
            "status":http.HTTPStatus.UNAUTHORIZED,
            "message":"Login Failed"
        }

    return {
        "status":http.HTTPStatus.OK,
        "user_id":user_id,
    }

@app.route("/predict", methods=['POST'])
def predict_handler():
    print("Got Predict request from: " + request.remote_addr)

    request_body = request.get_json()

    scan = uncompress_nparr(request_body['scan'])

    prediction = get_prediction(scan, classification_weights.model)
    prediction = prediction[0].item()
    prediction = round(prediction, 3)

    grad_cam = get_grad_cam(scan, classification_weights.model)
    grad = compress_nparr(grad_cam)

    generated_uuid = uuid.uuid4()

    ok = s3.upload_fileobj(grad, BUCKET_NAME, str(generated_uuid))
    if not ok:
        return {
            "status":http.HTTPStatus.INTERNAL_SERVER_ERROR,
            "message":"s3 error"
        }

    link = f's3://{BUCKET_NAME}/{generated_uuid}'

    ok = add_research(db,Research(user_id=request_body['user_id'],classification_bin_new=prediction,file_path=link))
    if not ok:
        return {
            "status":http.HTTPStatus.INTERNAL_SERVER_ERROR,
            "message":"s3 error"
        }

    if prediction < 0.6:
        return {
            "status": 200,
            "grad": grad,
            "prediction": prediction,
        }

    sygmentation_mask = get_heatmap(scan, classification_weights.model)

    return {
        "status": 200,
        "grad": grad,
        "prediction": prediction,
        "sygmentation_mask": sygmentation_mask,
    }

@app.route("/get_researches", methods=['POST'])
def login_handler():
    print("Got Predict request from: " + request.remote_addr)

    request_body = request.get_json()

    researches, ok = get_researches(db, request_body['user_id'])
    if not ok:
        return {
            "status":http.HTTPStatus.BAD_REQUEST,
            "message":f"User with {request_body['user_id']} user_id does not exist"
        }

    csv_string = "research_id,user_id,classification_bin_new,classification_bin_old,file_path,create_dt,update_dt\n"
    for research in researches:
        csv_string += f'{research.research_id},{research.user_id},{research.classification_bin_new},{research.classification_bin_old},{research.file_path},{research.create_dt},{research.update_dt}\n'


    return {
        "status":http.HTTPStatus.OK,
        "csv":csv_string,
    }