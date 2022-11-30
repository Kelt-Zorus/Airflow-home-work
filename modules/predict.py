import logging
import json
import dill
import os
import pandas as pd
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '.')

model_filename = sorted(os.listdir(f"{path}/data/models"))


def predict() -> None:
    car_id = []
    pred = []
    with open(os.path.join(f"{path}/data/models", model_filename[-1]), "rb") as file:
        model = dill.load(file)
    for filenames in os.listdir(f"{path}/data/test"):
        with open(os.path.join(f"{path}/data/test", filenames)) as name:
            car = json.load(name)
        df = pd.DataFrame(car, index=[0])
        y = model.predict(df)
        car_id.append(filenames[:-5])
        pred.append(y)
        pred_dict = {"car_id": car_id, "pred":pred}
    df_predict = pd.DataFrame.from_dict(pred_dict)

    csv_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_predict.to_csv(csv_filename, mode="a")

    logging.info(f'csv is saved as {csv_filename}')


if __name__ == '__main__':
    predict()
