#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import expm1

import pandas as pd

from flask import Flask, jsonify, request

from tensorflow import keras

from source.build_model import transform_data

app = Flask(__name__)

model = keras.models.load_model("nid_siem_model.h5")


@app.route("/", methods=["POST"])
def index():
    data = request.json
    print(data)

    df = pd.DataFrame(data, index=[0])

    df = transform_data(df)

    prediction = model.predict(df)

    predicted_connection = expm1(prediction.flatten()[0])

    return jsonify({"connection_type": str(predicted_connection)})


if __name__ == "__main__":
    app.run()
