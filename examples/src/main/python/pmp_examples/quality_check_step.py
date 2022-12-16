from pmp.decorators import *

from sagemaker.workflow.quality_check_step import (
    QualityCheckStep,
    DataQualityCheckConfig,
)

@with_inputs( arg('input_baseline_path', type_hint=str) )
@freestyle_step
def quality_check_step(input_baseline_path):
    QualityCheckStep(
            "GenerateDataChecks",
            quality_check_config = DataQualityCheckConfig(
                baseline_dataset = input_baseline_path,
                dataset_format = DatasetFormat.csv(header=True, output_columns_position="START"),
            ),
            check_job_config = CheckJobConfig(
                role=config.monitor.role,
                sagemaker_session=sagemaker_session
            ),
            skip_check=True,
            register_new_baseline=True,
        )