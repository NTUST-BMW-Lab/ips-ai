import argparse
from utils.loader import Loader
from models.random_forest import RandomForest
from models.svr import SVR
from models.dnn_regression import DNN_Regression
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
        help='Dataset path string',
        dest='data_path',
        default='data.csv',
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
        '--c',
        '--cleaned',
        help='identifier for data state (cleaned or not)',
        dest='cleaned',
        default=False,
        type=bool
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
        default=100,
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
    parser.add_argument(
        '--e',
        '--epochs',
        help='Number of iteration for deep learning model',
        dest='epochs',
        default=10,
        type=int
    )
    parser.add_argument(
        '--o',
        '--optimizer',
        help='optimizer function',
        dest='optimizer',
        default='adam',
        type=str
    )
    parser.add_argument(
        '--b',
        '--batch_size',
        help='batch size',
        dest='batch_size',
        default=8,
        type=int
    )
    parser.add_argument(
        '--vs',
        '--val_split',
        help='validation split value',
        dest='val_split',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--d',
        '--dropout',
        help='dropout value to decrease loss',
        dest='dropout',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--pow',
        '--power',
        help=' ap transmission power value enabler',
        dest='power',
        default=True,
        type=bool
    )

    args = parser.parse_args()

    dataset = Loader(
        data_path='./datas/' + args.data_path,
        preprocessor=args.preprocessor,
        cleaned=args.cleaned,
        frac=args.frac,
        no_val_rss=args.novalrss,
        floor=args.floor,
        test_size=args.test_size,
        power=args.power
    )

    training_data = dataset.training_data
    testing_data = dataset.testing_data
    waps = dataset.waps

    print(training_data)
    print(testing_data)
    print(waps)

    # if args.model == 'random_forest' or args.model == 'rf':
    #     train_model = RandomForest(random_state=args.random_state)
    # elif args.model == 'svr' or args.model == 'support_vector':
    #     train_model = SVR(random_state=args.random_state)
    # elif args.model == 'dnn':
    #     train_model = DNN_Regression(
    #         training_data=training_data, 
    #         testing_data=testing_data,
    #         random_state=args.random_state,
    #         preprocessor=args.preprocessor,
    #         batch_size=args.batch_size,
    #         epochs=args.epochs,
    #         optimizer=args.optimizer,
    #         validation_split=args.val_split,
    #         dropout=args.dropout,
    #         tx_power=args.power,
    #         patience=10,
    #         save_model=True
    #     )

    # train_model.train()
    # train_model.train(training_data.rss_scaled, training_data.labels)
    #predicted_coords = train_model.predict(testing_data.rss_scaled, testing_data.labels)
    #mse_val, r2_val = train_model.evaluate(testing_data.labels, predicted_coords)
    #save(model_name=model, mse_val=mse_val, r2_val=r2_val, predicted_coords=rf_predicted_coords, folder_dest='../evaluation')