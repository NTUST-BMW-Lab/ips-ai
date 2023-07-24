import argparse
from utils.loader import Loader
from models.random_forest import RandomForest
from models.svr import SVR
from utils.save_model import save

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--m',
        '--model',
        help='model to be used for training',
        dest='model',
        default='random_forest',
        type=str
    )
    parser.add_argument(
        '--dp',
        '--data_path',
        help='Data folder path',
        dest='data_path',
        default='../datas/',
        type=str
    )
    parser.add_argument(
        '--p',
        '--preprocessor',
        help='preprocessor method',
        dest='preprocessor',
        default='standard_scaler',
        type=str,
    )
    parser.add_argument(
        '--fr',
        '--frac',
        help='fraction of data to be sampled',
        dest='frac',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--n',
        '--novalrss',
        help='no rss value indicator',
        dest='novalrss',
        default=-100,
        type=int
    )
    parser.add_argument(
        '--f',
        '--floor',
        help='Indicates which floor to load',
        dest='floor',
        default=1,
        type=int
    )
    parser.add_argument(
        '--t',
        '--test_size',
        help='Test Size portion over Train Size',
        dest='test_size',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--r',
        '--random_state',
        help='random states',
        dest='random_state',
        default=0,
        type=int
    )

    args = parser.parse_args()

    dataset = Loader(
        path=args.data_path,
        preprocessor=args.preprocessor,
        frac=args.frac,
        no_val_rss=args.novalrss,
        floor=args.floor,
        test_size=args.test_size
    )

    training_data = dataset.training_data
    testing_data = dataset.testing_data

    if args.model == 'random_forest':
        train_model = RandomForest(random_state=args.random_state)
    elif args.model == 'svr':
        train_model = SVR(random_state=args.random_state)
    elif args.model == 'dnn':
        pass

    train_model.train(training_data.rss_scaled, training_data.labels)
    predicted_coords = train_model.predict(testing_data.rss_scaled, testing_data.labels)
    mse_val, r2_val = train_model.evaluate(testing_data.labels, predicted_coords)
    #save(model_name=model, mse_val=mse_val, r2_val=r2_val, predicted_coords=rf_predicted_coords, folder_dest='../evaluation')