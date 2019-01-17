import argparse

from train.lbf_regressor import LBFRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-directory", "-dataset", type=str, default="data/my_photos_14")
    parser.add_argument("--model-filename", "-model", type=str, default="trained_models/default_model.pkl")
    parser.add_argument("--n-landmarks", "-n-lm", type=int, default=14)
    parser.add_argument("--tree-depth", "-d", type=int, default=5)
    parser.add_argument("--n-estimators", "-n", type=int, default=300)
    parser.add_argument("--is-debug", "-debug", type=bool, default=False)
    parser.add_argument("--debug-size", "-ds", type=int, default=5)
    parser.add_argument("--image-format", "-f", type=str, default=".png")
    parser.add_argument("--n-jobs", "-jobs", type=int, default=2)
    args = parser.parse_args()
    print("test", args.n_landmarks)

    model = LBFRegressor(num_landmarks=args.n_landmarks, n_trees=args.n_estimators, tree_depth=args.tree_depth, n_jobs=args.n_jobs)
    model.load_data(args.dataset_directory, is_debug=args.is_debug, debug_size=args.debug_size, image_format=args.image_format)
    model.train()
    model.save_model(args.model_filename)

