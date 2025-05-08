import matplotlib.pyplot as plt
import pandas as pd


def plot_speed_histogram(df_speed_kmh: pd.DataFrame, simulation_id: str, bins: int = 30) -> None:
    """Plot a histogram of vehicle speeds in km/h.

    Args:
        df (pd.DataFrame): DataFrame containing vehicle speed data in km/h.
        simulation_id (str): Simulation ID for the plot title.
        bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(df_speed_kmh, bins=bins)
    plt.title(f"Speed Histogram - {simulation_id}")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Count")
    plt.show()


def plot_average_speed_and_vehicle_count_per_second(
    df_second: pd.DataFrame,
    df_average_speed_kmh_per_second: pd.DataFrame,
    df_vehicle_count_per_second: pd.DataFrame,
    simulation_id: str,
) -> None:
    """Plot average speed and vehicle count on a per-second basis.

    Args:
        df_second (pd.DataFrame): DataFrame containing second-wise data.
        df_average_speed_kmh_per_second (pd.DataFrame): DataFrame containing average speed per hour in km/h.
        df_vehicle_count_per_second (pd.DataFrame): DataFrame containing vehicle count per second.
        simulation_id (str): Simulation ID for the plot title.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_second, df_average_speed_kmh_per_second, label="Average Speed (km/h)")
    ax1.set_xlabel("Second")
    ax1.set_ylabel("Average Speed (km/h)")

    ax2 = ax1.twinx()
    ax2.plot(df_second, df_vehicle_count_per_second, label="Vehicle Count", color="green")
    ax2.set_ylabel("Count")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="upper right")
    plt.title(f"Average Speed & Vehicle Count (Per Second) - {simulation_id}")
    plt.show()


def plot_average_speed_and_traffic_generation_period_per_hour(
    df_hour: pd.DataFrame,
    df_average_speed_kmh_per_hour: pd.DataFrame,
    traffic_generation_periods: list,
    simulation_id: str,
) -> None:
    """Plot average speed and traffic generation period on a per-hour basis.

    Args:
        df_hour (pd.DataFrame): DataFrame containing hour-wise data.
        df_average_speed_kmh_per_hour (pd.DataFrame): DataFrame containing average speed per hour in km/h.
        traffic_generation_periods (list): List of traffic generation periods in seconds.
        simulation_id (str): Simulation ID for the plot title.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_hour, df_average_speed_kmh_per_hour, marker="o", label="Average Speed (km/h)")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Average Speed (km/h)")

    ax2 = ax1.twinx()
    ax2.plot(df_hour, traffic_generation_periods, marker="o", label="Traffic Generation Period (s)", color="orange")
    ax2.set_ylabel("Traffic Generation Period (s)")

    lines, labels = ax1.get_lines() + ax2.get_lines(), [l.get_label() for l in ax1.get_lines() + ax2.get_lines()]
    fig.legend(lines, labels, loc="upper right")
    plt.title(f"Average Speed & Traffic Generation Period (Per Hour) - {simulation_id}")
    plt.show()
