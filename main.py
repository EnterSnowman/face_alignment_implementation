import argparse

from train.lbf_regressor import LBFRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-directory", "-dataset", type=str, default="data/my_photos_14")
    parser.add_argument("--model-name", "-model", type=str, default="default_model_2")
    parser.add_argument("--trained-models-dir", "-model-dir", type=str, default="trained_models")
    parser.add_argument("--config-file", "-config", type=str, default=None)

    parser.add_argument("--n-landmarks", "-n-lm", type=int, default=14)
    parser.add_argument("--tree-depth", "-d", type=int, default=5)
    parser.add_argument("--n-estimators", "-n", type=int, default=300)
    parser.add_argument("--is-debug", "-debug", type=bool, default=False)
    parser.add_argument("--debug-size", "-ds", type=int, default=5)
    parser.add_argument("--image-format", "-f", type=str, default=".png")
    parser.add_argument("--n-jobs", "-jobs", type=int, default=-1)
    args = parser.parse_args()
    print("test", args.n_landmarks)
    config_file = "trained_models/default_model_2/default_model_2_conf.txt"
    # config_file = None

    model = LBFRegressor(num_landmarks=args.n_landmarks, n_trees=args.n_estimators, tree_depth=args.tree_depth,
                         n_jobs=args.n_jobs, model_name=args.model_name, config_file=config_file)
    model.load_data(args.dataset_directory, is_debug=True, debug_size=args.debug_size, image_format=args.image_format)
    model.train()
    # model.save_model(args.model_filename)


