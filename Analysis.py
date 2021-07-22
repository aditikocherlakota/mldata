import argparse
import numpy as np
import bz2
import pickle
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import yaml

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_one_data_files(files):
    for file in files:
        data = {}
        with bz2.BZ2File(file, "rb") as f:
            tmp_data = pickle.load(f)
        data["yaml"] = tmp_data[0]
        data["value_of_permuted_feild"] = tmp_data[1]
        data["coordinate"] = tmp_data[2]
        data["timeVSvmem"] = tmp_data[3]

        yield file, data


def massage_the_data():
    global args
    files = [f for f in glob.glob(args.test_path + "/" + args.data_path + "/*.pkl")]
    load_andler = load_one_data_files(files=files)

    # load data
    dataPoints = list()
    dataPoint_file = list()
    for file, dataPoint in tqdm(load_andler, total=len(files)):
        if args.save_frames_from_begining:
            tmp_d = dataPoint["timeVSvmem"][: args.number_of_frames_to_analyse, :]
        else:
            tmp_d = dataPoint["timeVSvmem"][-args.number_of_frames_to_analyse :, :]
        dataPoints.append(tmp_d)
        dataPoint_file.append(file)

    os.makedirs(args.test_path + "/" + args.analysis_path + "/", exist_ok=True)
    with bz2.BZ2File(
        args.test_path
        + "/"
        + args.analysis_path
        + "/dataPoints_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump([dataPoints, dataPoint_file], f)

    return dataPoints, dataPoint_file


def normalize_data(dataPoints):
    # find the maximum number of cells
    min_shape = max_shape = dataPoints[0].shape[1]
    time_shape = dataPoints[0].shape[0]
    for dataPoint in dataPoints:
        if max_shape < dataPoint.shape[1]:
            max_shape = dataPoint.shape[1]

        if min_shape > dataPoint.shape[1]:
            min_shape = dataPoint.shape[1]

    from scipy import interpolate

    for d in tqdm(range(len(dataPoints))):
        new_data = np.zeros((time_shape, max_shape), dtype=float)
        t_line = np.arange(0, dataPoints[d].shape[1], 1)
        t_line_new = np.arange(0, max_shape, 1)
        for i in range(dataPoints[d].shape[0]):
            s = interpolate.InterpolatedUnivariateSpline(t_line, dataPoints[d][i, :])
            new_data[i, :] = s(t_line_new)
        dataPoints[d] = new_data.reshape(-1)

    dataPoints = np.array(dataPoints, dtype=float)

    # normalize the data -- because we dont care how high the voltage is, we only care the magnatude
    # of the change in the voltage, normalization is the sensable choice, because it normalize the data
    # between the min and max of the sample and not the all dataset.
    dataPoints = Normalizer().fit(dataPoints).transform(dataPoints)

    #reshape back to the original shape
    dataPoints = dataPoints.reshape(-1, time_shape, max_shape)
    
    os.makedirs(
        args.test_path + "/" + args.analysis_path + "/" + args.ml_path + "/",
        exist_ok=True,
    )
    with bz2.BZ2File(
        args.test_path
        + "/"
        + args.analysis_path
        + "/dataPoints_normalize_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(dataPoints, f)

    return dataPoints


def analysis_entropy_witout_time(data, data_files):
    def entropy(vac):
        # https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated
        marg = (
            np.histogramdd(np.ravel(vac), bins=65536)[0] / vac.size
        )  # 65536 = 2^16 = number of possible values in the voltagee
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        entropy = -np.sum(np.multiply(marg, np.log2(marg)))
        return entropy

    global args

    variance = list()
    for data_point in tqdm(data):
        variance.append(entropy(data_point.reshape(-1)))

    # save BETSE Min, Max, Mid datapoints in the container
    return_data_points = dict()
    return_data_points["min"] = min(variance)
    f_loc = data_files[np.where(return_data_points["min"] == variance)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["minDataPoint"] = pickle.load(f)

    return_data_points["max"] = max(variance)
    f_loc = data_files[np.where(return_data_points["max"] == variance)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["maxDataPoint"] = pickle.load(f)

    return_data_points["mid"] = variance[
        np.where(
            min(abs(np.median(variance) - variance))
            == abs(np.median(variance) - variance)
        )[0][0]
    ]
    f_loc = data_files[np.where(return_data_points["mid"] == variance)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["midDataPoint"] = pickle.load(f)

    # --- Save container
    os.makedirs(
        args.test_path + "/" + args.analysis_path + "/" + args.delta_path + "/",
        exist_ok=True,
    )
    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/entropy_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".csv",
        "wt",
    ) as f:
        np.savetxt(
            f,
            np.array(variance, dtype=float),
            delimiter=",",
        )

    with bz2.BZ2File(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/entropy_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(return_data_points, f)

    # --- Extract betse config files from datapoints
    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/minDataPoint_entropy_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["minDataPoint"][0], stream=f)

    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/maxDataPoint_entropy_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["maxDataPoint"][0], stream=f)

    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/medDataPoint_entropy_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["midDataPoint"][0], stream=f)

    return variance, return_data_points


def analysis_flactuations_witout_time(data, data_files):
    global args

    variance = list()
    for data_point in tqdm(data):
        variance.append(np.var(data_point))

    # save BETSE Min, Max, Mid datapoints in the container
    return_data_points = dict()
    return_data_points["min"] = min(variance)
    f_loc = data_files[np.where(return_data_points["min"] == variance)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["minDataPoint"] = pickle.load(f)

    return_data_points["max"] = max(variance)
    f_loc = data_files[np.where(return_data_points["max"] == variance)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["maxDataPoint"] = pickle.load(f)

    return_data_points["mid"] = variance[
        np.where(
            min(abs(np.median(variance) - variance))
            == abs(np.median(variance) - variance)
        )[0][0]
    ]
    f_loc = data_files[np.where(return_data_points["mid"] == variance)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["midDataPoint"] = pickle.load(f)

    # --- Save container
    os.makedirs(
        args.test_path + "/" + args.analysis_path + "/" + args.delta_path + "/",
        exist_ok=True,
    )
    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/variance_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".csv",
        "wt",
    ) as f:
        np.savetxt(
            f,
            np.array(variance, dtype=float),
            delimiter=",",
        )

    with bz2.BZ2File(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/variance_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(return_data_points, f)

    # --- Extract betse config files from datapoints
    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/minDataPoint_variance_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["minDataPoint"][0], stream=f)

    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/maxDataPoint_variance_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["maxDataPoint"][0], stream=f)

    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/medDataPoint_variance_wt_time_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["midDataPoint"][0], stream=f)

    return variance, return_data_points


def analysis_flactuations_in_time(data, data_files):
    global args

    delta = list()
    for data_point in tqdm(data):
        tmp_delta = 0
        for t in range(1, data_point.shape[0]):
            tmp_delta += abs(data_point[t - 1, :] - data_point[t, :]).sum()
        delta.append(tmp_delta)

    # save BETSE Min, Max, Mid datapoints in the container
    return_data_points = dict()
    return_data_points["min"] = min(delta)
    f_loc = data_files[np.where(return_data_points["min"] == delta)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["minDataPoint"] = pickle.load(f)

    return_data_points["max"] = max(delta)
    f_loc = data_files[np.where(return_data_points["max"] == delta)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["maxDataPoint"] = pickle.load(f)

    return_data_points["mid"] = delta[
        np.where(min(abs(np.median(delta) - delta)) == abs(np.median(delta) - delta))[
            0
        ][0]
    ]
    f_loc = data_files[np.where(return_data_points["mid"] == delta)[0][0]]
    with bz2.BZ2File(f_loc, "rb") as f:
        return_data_points["midDataPoint"] = pickle.load(f)

    # --- Save container
    os.makedirs(
        args.test_path + "/" + args.analysis_path + "/" + args.delta_path + "/",
        exist_ok=True,
    )
    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/delta_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".csv",
        "wt",
    ) as f:
        np.savetxt(
            f,
            np.array(delta, dtype=float),
            delimiter=",",
        )

    with bz2.BZ2File(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/dataPoints_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(return_data_points, f)

    # --- Extract betse config files from datapoints
    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/minDataPoint_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["minDataPoint"][0], stream=f)

    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/maxDataPoint_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["maxDataPoint"][0], stream=f)

    with open(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.delta_path
        + "/medDataPoint_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(args.save_frames_from_begining)
        + ".yaml",
        "wt",
    ) as f:
        yaml.dump(data=return_data_points["midDataPoint"][0], stream=f)

    return delta, return_data_points


def plot_scatter(X, title=None):
    fig = plt.figure()

    if X.shape[1] == 2:  # 2D
        ax = plt.subplot(111)
        ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
    elif X.shape[1] == 3:  # 3D
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5)

    if title is not None:
        plt.title(title)


def analysis_with_tSNE(dataPoints, name='', componets=3):
    
    if len(dataPoints.shape) > 2:
        dataPoints = dataPoints.reshape(dataPoints.shape[0], -1)
    
    # tSNE
    tSNE = TSNE(
        n_components=componets
    )  # , method='exact', perplexity= 50, n_iter=1000,
    Y = tSNE.fit_transform(dataPoints)

    os.makedirs(
        args.test_path + "/" + args.analysis_path + "/" + args.ml_path + "/",
        exist_ok=True,
    )
    with bz2.BZ2File(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.ml_path
        + "/tSNE_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(name)
        + "_"
        + str(args.save_frames_from_begining)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(Y, f)

    return Y


def analysis_with_PCA(dataPoints, name='', componets=3):
        
    if len(dataPoints.shape) > 2:
        dataPoints = dataPoints.reshape(dataPoints.shape[0], -1)
    
    # PCA
    pca = PCA(n_components=componets)
    Y = pca.fit_transform(dataPoints)

    os.makedirs(
        args.test_path + "/" + args.analysis_path + "/" + args.ml_path + "/",
        exist_ok=True,
    )
    with bz2.BZ2File(
        args.test_path
        + "/"
        + args.analysis_path
        + "/"
        + args.ml_path
        + "/PCA_"
        + str(args.number_of_frames_to_analyse)
        + "_"
        + str(name)
        + "_"
        + str(args.save_frames_from_begining)
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(Y, f)

    return Y


def main():
    global args
    print(f"\nCWD = {os.getcwd()}\n")

    # #---- load data
    data, data_files = massage_the_data()

    # Normalize data
    file = args.test_path + '/' + args.analysis_path + '/dataPoints_' + str(args.number_of_frames_to_analyse) + '_' + str(args.save_frames_from_begining) + '.pkl'
    with bz2.BZ2File(file, 'rb') as f:
        data, data_files = pickle.load(f)

    data = normalize_data(dataPoints=data)

    # -------------------------------
    # Delta between frames, data with in frames, entropy with in frames
    # file = (
    #     args.test_path
    #     + "/"
    #     + args.analysis_path
    #     + "/dataPoints_"
    #     + str(args.number_of_frames_to_analyse)
    #     + "_"
    #     + str(args.save_frames_from_begining)
    #     + ".pkl"
    # )
    # with bz2.BZ2File(file, "rb") as f:
    #     data, data_files = pickle.load(f)

    _, _ = analysis_flactuations_in_time(data, data_files)
    _, _ = analysis_flactuations_witout_time(data, data_files)
    _, _ = analysis_entropy_witout_time(data, data_files)

    # # -------------------------------
    # # Delta Histogram
    # file = args.test_path + '/' + args.analysis_path + "/" + args.delta_path + '/delta_' + str(args.number_of_frames_to_analyse) + '_' + str(args.save_frames_from_begining) + '.csv'
    # data = np.loadtxt(file, delimiter=',')

    # num,bin = np.histogram(data, bins=100)
    # n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
    # alpha=0.7, rwidth=0.85)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Complexity - delta')
    # plt.ylabel('Frequency')
    # plt.title('Complexity patterns in random parameter search')
    # maxfreq = n.max()
    # # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.show()
    # # -------------------------------

    # # ---- tSNE
    # with bz2.BZ2File(
    #     args.test_path
    #     + "/"
    #     + args.analysis_path
    #     + "/dataPoints_normalize_"
    #     + str(args.number_of_frames_to_analyse)
    #     + "_"
    #     + str(args.save_frames_from_begining)
    #     + ".pkl",
    #     "rb",
    # ) as f:
    #     data = pickle.load(f)

    tSNE = analysis_with_tSNE(data, name='normalize', componets=3)

    # with bz2.BZ2File(args.test_path + '/' + args.analysis_path + '/' + args.ml_path + '/tSNE_' + str(args.number_of_frames_to_analyse) + '_' + str(args.save_frames_from_begining) + '.pkl', 'rb') as f:
    # tSNE = pickle.load(f)

    # plot_scatter(tSNE)
    # plt.show()

    # # ----- PCA
    # with bz2.BZ2File(
    #     args.test_path
    #     + "/"
    #     + args.analysis_path
    #     + "/dataPoints_normalize_"
    #     + str(args.number_of_frames_to_analyse)
    #     + "_"
    #     + str(args.save_frames_from_begining)
    #     + ".pkl",
    #     "rb",
    # ) as f:
    #     data = pickle.load(f)

    pca = analysis_with_PCA(data, name='normalize', componets=3)

    # with bz2.BZ2File(args.test_path + '/' + args.analysis_path + '/' + args.ml_path + '/PCA_' + str(args.number_of_frames_to_analyse) + '_' + str(args.save_frames_from_begining) + '.pkl', 'rb') as f:
    # pca = pickle.load(f)

    # plot_scatter(pca)
    # plt.show()

    print("Main Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="data_logs")
    parser.add_argument("--analysis_path", type=str, default="Analysis")
    parser.add_argument("--delta_path", type=str, default="Delta")
    parser.add_argument("--ml_path", type=str, default="ML")
    parser.add_argument("--data_report_path", type=str, default="DeltaReport")
    parser.add_argument("--number_of_frames_to_analyse", type=int, default=0)
    parser.add_argument("--save_frames_from_begining", type=str2bool, default=False)

    args = parser.parse_args()

    main()
