# Prevalence Bias Research

## Catalogue Explanation

- book_rec:NAGHIAEI M, RAHMANI HosseinA, DEHGHAN M. The Unfairness of Popularity Bias in Book Recommendation[J]. The source code of the paper, which was used to process the data from this paper for Book-Crossing
- dataset: integer-converted dataset
- Debias:Recommendation algorithms and custom models, including data partitioning, training models, testing models, collecting results, etc.
- algorithm: contains Baseline and custom models.
- log: record the training results of each model, in the format of model_name-dataset
- process_data: store the intermediate results of dataset division
- result: collection of experimental results
- temp_result: contains the results of each model on each dataset individually
- aggregation.xlsx: aggregated results
- utils: early stop mechanism used for training.
- bias_visual.py: visualisation of experimental results for prevalence bias
- filter_result_to_excel.py: aggregated all experimental results to excel
- metrics.py: defined evaluation metrics
- run.py: run individual models
- run_all.py: run all models.
- scan_all_results.py: collect results of separate experiments for each model on each dataset
- img:image to store for data analysis
- naive_data:unprocessed dataset
- dataset_statistics.py: dataset statistics, including number of users, number of items, total number of interactions, and sparseness
- rec_data_convert_by_id.py: converts user ids and project ids in the dataset to integers
- requirements.txt: required dependency packages
- visual.py: visualisation of the dataset analysis
