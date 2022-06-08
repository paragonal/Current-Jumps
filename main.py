import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import permutations


def plot_IV_curve_simple(file, sheet_name, axs=None, silent=False):
    """
    :param file: Name of the file to read from
    :param sheet_name: Name of the sheet in the file
    :param axs: Optional argument with axs to plot on
    """
    df = pd.read_excel(file, sheet_name=sheet_name)
    df = df.query("Vsd > 0")
    if silent:
        return df['Vsd'], df['Current']

    if axs == None:
        axs = plt.axes()

    axs.plot(df['Vsd'], df['Current'])
    axs.set_xlabel("Voltage")
    axs.set_ylabel("Current")
    axs.set_title("Voltage-Current curve for {}".format(sheet_name))
    return df['Vsd'], df['Current']


def plot_pairs(file, sheet_name, pairs):
    """
    :param file: Name of the file to read from
    :param sheet_name: Name of the sheet in the file
    :param pairs: List of 2-tuples to plot against each other
    """
    df = pd.read_excel(file, sheet_name=sheet_name)
    fig, axes = plt.subplots(len(pairs), figsize=(10, 6))
    for i, p in enumerate(pairs):
        axes[i].plot(df[p[0]], df[p[1]])
        axes[i].set_xlabel(p[0])
        axes[i].set_ylabel(p[1])
    plt.tight_layout()


def plot_pair(file, sheet_name, p1, p2, silent=False):
    """
    :param file: Name of the file to read from
    :param sheet_name: Name of the sheet in the file
    :param pairs: Quantities to plot against eachother
    """
    pair = (p1, p2)
    df = pd.read_excel(file, sheet_name=sheet_name)
    if silent:
        return df[pair[0]], df[pair[1]]
    plt.plot(df[pair[0]], df[pair[1]])
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    return df[pair[0]], df[pair[1]]


def plot_all_pairs(file, sheet_name, column_titles):
    """
    :param file: Name of the file to read from
    :param sheet_name: Name of the sheet in the file
    :param column_titles : All columns to plot against eachother
    """
    df = pd.read_excel(file, sheet_name=sheet_name)
    fig, axes = plt.subplots(len(column_titles), len(column_titles), figsize=(12, 10))
    for i, p in enumerate(column_titles):
        for j, p2 in enumerate(column_titles):
            axes[i, j].plot(df[p], df[p2])
            axes[i, j].set_xlabel(p)
            axes[i, j].set_ylabel(p2)
    plt.tight_layout()


voltages, currents = plot_IV_curve_simple("KR2B9IVcurv1.xlsx", sheet_name="H_R1B9_IVcurve_4", silent=True)


def fit_fun(x, a, b, c, d, f):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + f


# Poly interp to get conductance as func of voltage

p, _ = curve_fit(fit_fun, voltages, currents)
p = list(p)
outputs = np.polyval(p, voltages)

derivs = np.polyder(p)
deriv_outputs = np.polyval(derivs, voltages)

fig, axs = plt.subplots(2, figsize=(10, 6))
axs[0].set_title("R1B9 IV Curves")
axs[1].set_xlabel("Voltage")
axs[0].plot(voltages, currents, label="Observed")
axs[0].plot(voltages, outputs, label="Fitted")
axs[0].legend()
axs[0].set_ylabel("Current")

axs[1].plot(voltages, deriv_outputs)
axs[1].set_ylabel("Conductance")

# chisq = (np.array(currents)-outputs)**2/abs(np.array(currents))
# print("Values of coeffs", p)
# print("chisq", sum(chisq))
plt.show()

# ## Get conductance as func of temp
# file = "R1B9_R-vs-T.xlsx"
# sheet_name = "R_vs_T_R1B92_0T_5mV"
# df = pd.read_excel(file, sheet_name=sheet_name)
# temp, current, voltage = df["Temp"], df["Current"], df["Voltage"]
# conductance = current / voltage
# plt.plot(temp, conductance)
# plt.show()
