from pmp.decorators import *
import logging
import pandas as pd
from sagemaker.sklearn import SKLearn
from sklearn.model_selection import train_test_split

from sagemaker.processing import (
    FrameworkProcessor,
)

logging.basicConfig(level=logging.INFO)

DEFAULT_DATA_FILE_NAME = "data.csv"
DEFAULT_HEADER_FILE_NAME = "header.json"
FEATURIZER_FILE_NAME = "column_transformer.model"

ID_COLUMN = "ID"
LABEL_COLUMN = "rings"

# Since we get a headerless CSV file we specify the column names here.
DATA_COLUMN_NAMES = [
    ID_COLUMN,
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]

LABEL_COLUMN_NAMES = [
    ID_COLUMN,
    LABEL_COLUMN
]

DATA_COLUMN_DTYPES = {
    ID_COLUMN : np.int64,
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}

LABEL_COLUMN_DTYPES = {
    ID_COLUMN : np.int64,
    LABEL_COLUMN: np.float64
}


def save_to_csv(df, output_path, header=False, index=True):
    logger = logging.getLogger(__name__)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info("Writing to %s", output_path / DEFAULT_DATA_FILE_NAME)
    df.to_csv(output_path / DEFAULT_DATA_FILE_NAME, header=header, index=index)

def read_data_csv(fn, header=None):
    return pd.read_csv(
        Path(fn, DEFAULT_DATA_FILE_NAME),
        header=header,
        names=DATA_COLUMN_NAMES,
        dtype=DATA_COLUMN_DTYPES,
        index_col=ID_COLUMN,
    )

def split_dataset(data_df: pd.DataFrame, label_df: pd.DataFrame, train_part):
    (
        train_data_df,
        test_data_df,
        train_label_df,
        test_label_df,
    ) = train_test_split(
        data_df, label_df, test_size=1 - train_part, shuffle=True
    )
    return train_data_df, train_label_df, test_data_df, test_label_df

def create_processor(sagemaker_session, config) -> FrameworkProcessor:
    return FrameworkProcessor(
        estimator_cls=SKLearn,
        # TODO upgrade to 1.0
        framework_version="0.23-1",
        role=config.featurizing.role,
        instance_count=config.featurizing.instance_count,
        instance_type=config.featurizing.instance_type,
        sagemaker_session=sagemaker_session,
    )


@with_inputs( arg('input_data_path', type_hint=str),
              arg('input_label_path', type_hint=str) )
@with_outputs( arg('output_train_dataset_data', type_hint=str),
               arg('output_train_dataset_labels', type_hint=str),
               arg('output_test_dataset_data', type_hint=str),
               arg('output_test_dataset_labels', type_hint=str) )
@with_parameters( arg('train_part', type_hint=float, default=0.8) )
@on_framework_processor(create_processor)
def first_step_split_example(
                        input_data_path,
                        input_label_path,
                        output_train_dataset_data,
                        output_train_dataset_labels,
                        output_test_dataset_data,
                        output_test_dataset_labels,
                        train_part):
   data_df = read_data_csv(input_data_path, header=0)
   label_df = read_label_csv(input_label_path, header=0)
   train_data_df, train_label_df, test_data_df, test_label_df = split_dataset(
       data_df, label_df, train_part
   )
   save_to_csv(train_data_df, output_train_dataset_data)
   save_to_csv(train_label_df, output_train_dataset_labels)
   save_to_csv(test_data_df, output_test_dataset_data)
   save_to_csv(test_label_df, output_test_dataset_labels)

   result = data_df.first().asDict(True)

   return {"result": result}
