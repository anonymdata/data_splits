from datasets import dunnhumby, tafeng, movielens


def load(config):
    """
    Loading dataset
    """
    root_dir = config["root_dir"]
    if "test_percent" not in config:
        test_percent = None
    else:
        test_percent = config["test_percent"]
    if config["dataset"] == "dunnhumby":
        if config["data_split"] == "temporal":
            train_df, validate_df, test_df = dunnhumby.load_temporal(root_dir=root_dir)
        elif config["data_split"] == "leave_one_item":
            train_df, validate_df, test_df = dunnhumby.load_leave_one_item(
                root_dir=root_dir
            )
        elif config["data_split"] == "leave_one_basket":
            train_df, validate_df, test_df = dunnhumby.load_leave_one_basket(
                root_dir=root_dir
            )
    elif config["dataset"] == "tafeng":
        if config["data_split"] == "temporal":
            train_df, validate_df, test_df = tafeng.load_temporal(
                root_dir=root_dir, test_percent=test_percent
            )
        elif config["data_split"] == "leave_one_item":
            train_df, validate_df, test_df = tafeng.load_leave_one_item(
                root_dir=root_dir
            )
        elif config["data_split"] == "leave_one_basket":
            train_df, validate_df, test_df = tafeng.load_leave_one_basket(
                root_dir=root_dir
            )
    elif config["dataset"] == "movielens" or config["dataset"] == "ml-1m":
        if config["data_split"] == "temporal":
            train_df, validate_df, test_df = movielens.load_temporal(root_dir=root_dir)
        elif (
            config["data_split"] == "leave_one_item"
            or config["data_split"] == "leave_one_out"
        ):
            train_df, validate_df, test_df = movielens.load_leave_one_out(
                root_dir=root_dir
            )
    else:
        print("get the wrong dataset or data_split.")

    return train_df, validate_df, test_df