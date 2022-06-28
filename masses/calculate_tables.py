import pandas as pd


def save_monoisotopic_masses() -> None:
    """
    Calculates table of monoisotopic masses from isotope abundance table
    :return:
    """
    df = pd.read_csv("isotope_abundance.csv", sep=";")
    idx = df.groupby(['element'])['abundance'].transform(max) == df['abundance']
    res = df[idx]
    res.to_csv("monoisotopic_masses.csv", sep=";", index=False)


def save_average_masses():
    """
    Calculates average mass of each element
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


def save_isotopic_masses() -> None:
    """
    Calculates table of isotopic masses from isotope abundance table
    For example, element: 13C, mass 13.003355, abundance: 0.011
    :return:
    """
    df = pd.read_csv("isotope_abundance.csv", sep=';')
    df["abundance"] = df["abundance"] / 100.0

    res = []
    for index, row in df.iterrows():
        row = dict(row)
        row['element'] = f"{round(row['mass'])}{row['element']}"
        res.append(row)

    pd.DataFrame(res).to_csv("isotopic_masses.csv", sep=";", index=False)

def func(x):
    d = x['element'] + '_' + str(x['isotop'])
    return d

def save_with_istopes_label()-> None:
    '''
    
    '''
    df = pd.read_csv("isotope_abundance.csv", sep=';')
    df['isotop'] = df['mass'].astype(int)
    df['element_isotop'] = df.apply(func, axis=1)
    df.to_csv('element_table.csv')


if __name__ == "__main__":
    save_monoisotopic_masses()
    save_average_masses()
    save_isotopic_masses()
    save_with_istopes_label()
