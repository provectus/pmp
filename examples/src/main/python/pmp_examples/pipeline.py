from pmp.decorators import *


@with_name( arg('pipeline_name', type_hint=str) )
@with_parameters( arg('input_s3_uri', type_hint=str) )
@with_steps( first_step_split_example,
             second_step_generate_constraints_for_data_quality_monitoring,
             quality_check_step )
@on_pipeline
def pipeline( pipeline,
              first_step_split_example,
              second_step_generate_constraints_for_data_quality_monitoring,
              quality_check_step ):

    second_step_generate_constraints_for_data_quality_monitoring.inputs['input_data_path'] = first_step_split_example.outputs['train-dataset-data']
    quality_check_step.inputs['input_baseline_path'] = second_step_generate_constraints_for_data_quality_monitoring.outputs['output_path']

    pipeline.run

