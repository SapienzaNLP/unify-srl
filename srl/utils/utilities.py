def check_argument_dataset(dataset_type, dataset_combination, select_dataset):
    type_dataset_combination = type(dataset_combination)
    assert isinstance(dataset_type, type_dataset_combination)
    if isinstance(dataset_type, type('')):  # if is a string
        assert dataset_type in select_dataset.keys()
        assert type_dataset_combination in select_dataset['dev'].keys()

