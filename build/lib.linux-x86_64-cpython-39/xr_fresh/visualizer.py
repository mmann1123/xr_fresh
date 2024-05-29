import geowombat as gw
import matplotlib.pyplot as plt


__all__ = ["plot_interpolated_actual"]


# visualize interpolation
def open_files(predict: str, actual: str):
    with gw.open(predict) as predict:
        with gw.open(actual, stack_dim="band") as actual:
            return predict, actual


def sample_data(predict, actual, n=20):
    df1 = gw.sample(predict, n=20).dropna().reset_index(drop=True)
    df2 = gw.extract(actual, df1[["point", "geometry"]])
    return df1, df2


def plot_data(df1, df2):
    fig, ax = plt.subplots(figsize=(10, 6))
    time_points = list(range(1, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df1)))

    for idx, row in df2.iterrows():
        ax.scatter(
            time_points,
            row[time_points],
            color=colors[idx],
            label=f"actual, Point {row['point']}",
            linestyle="-",
        )

    for idx, row in df1.iterrows():
        ax.plot(
            time_points,
            row[time_points],
            color=colors[idx],
            label=f"predicted, Point {row['point']}",
            linestyle="--",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series Comparison Between Predicted and Actual Values")
    plt.show()


def plot_interpolated_actual(
    interpolated_stack: str, original_image_list: list, samples: int = 20
):
    """Plots the interpolated and actual values for a given time series.

    Args:
        interpolated_stack (str): multiband stack of images representing interpolated time series. Defaults to None.
        original_image_list (list): list of files used in interpolation. Defaults to None.
        samples (int, optional): number of random points to compare time series. Defaults to 20.
    """
    predict, actual = open_files(
        interpolated_stack,
        original_image_list,
    )
    df1, df2 = sample_data(predict, actual, n=samples)
    plot_data(df1, df2)
