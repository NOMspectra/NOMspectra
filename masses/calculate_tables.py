import pandas as pd


def save_monoisotopic_masses() -> None:
    """
    Calculate table of monoisotopic masses from isotope abundance table
    :return:
    """
    df = pd.read_csv("isotope_abundance.csv", sep=";")
    idx = df.groupby(['element'])['abundance'].transform(max) == df['abundance']
    res = df[idx]
    res.to_csv("monoisotopic_masses.csv", sep=";", index=False)


def save_average_masses():
    """
    Calculate average mass of each element
    :return:
    """
    df = pd.read_csv("isotope_abundance.csv", sep=";")
    df["abundance"] = df["abundance"] / 100.0

    res = pd.DataFrame()
    for element, table in df.groupby(by="element"):
        mass = (table["mass"] * table["abundance"]).sum()
        res = res.append({
                "element": element,
                "mass": mass
            },
            ignore_index=True
        )
    res.to_csv("average_masses.csv", sep=";", index=False)


if __name__ == "__main__":
    save_monoisotopic_masses()
    save_average_masses()
