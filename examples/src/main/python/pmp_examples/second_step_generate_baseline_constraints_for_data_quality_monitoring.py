from pmp.decorators import *

FORMAT_CSV_FULL = lambda df, output_path: save_to_csv(df, output_path, header=True, index=True)
FORMAT_CSV_XGBOOST = lambda df, output_path: save_for_xgboost(df, output_path)
FORMAT_CSV_DATAQA = lambda df, output_path: save_to_csv(df, output_path, header=True, index=False)

@with_args( arg('input_data_path', type_hint=str),
            arg('output_path'))
@on_framework_processor(create_processor)
def second_step_generate_baseline_constraints_for_data_quality_monitoring( input_data_path, output_path ):
    data_df = read_data_csv(input_data_path)
    data_df = data_df.reset_index(drop=True)

    FORMAT_CSV_DATAQA(data_df, output_path)