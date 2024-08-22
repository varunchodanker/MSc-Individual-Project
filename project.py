import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def write_ids(df, idname, filename):
    """ Writes the unique, non-NaN instances of the indicated identifier, within the 
    indicated dataframe, on separate lines of a new file, whose filename should 
    be specified.

    Args:
        df: Dataframe.
        idname: Column name of the identifier with respect to the dataframe.
        filename: The name to use for the newly created file.
    """
    with open(filename, "w") as fh:
        for idval in df[idname].dropna().unique():
            fh.write(f"{idval}\n")


def read_emissions(filename):
    return pd.read_csv(
        filename, 
        dtype={"companyid": "str", "gvkey": "str"}, #identifiers
        parse_dates=["periodenddate"], 
    )


def write_emission_companyids(filename):
    """  """

    emissions = read_emissions(filename)
    write_ids(emissions, "companyid", "companyids.txt")


def load_emissions(emissions_filename, jointable_filename):
    """ Loads the emissions dataset and performs the appropriate pre-processing 
    for linking. """

    emissions = read_emissions(filename)

    cid_gvkey_mappings = pd.read_csv(jointable_filename)
    # consider 1 to 1 mappings to prevent redudancies in emissions information for gvkey
    cid_gvkey_mappings = cid_gvkey_mappings[
        (cid_gvkey_mappings["startdate"] == "B") & (cid_gvkey_mappings["enddate"] == "E")
    ]

    # filter down emissions
    emissions = emissions[pd.notnull(emissions["gvkey"])]
    # non-null allows us to finally change it to integer
    emissions["gvkey"] = emissions["gvkey"].astype(int)
    # 1 to 1 to prevent redundant/conflicting emissions information
    emissions = emissions[
        np.isin(emissions["gvkey"], cid_gvkey_mappings["gvkey"])
    ]
    # remove rows with missing values
    emissions = emissions.dropna()

    # finalise the dataframe, ready for linking
    emissions = emissions.reset_index(drop=True)
    emissions = emissions.set_index(
        ["gvkey", "fiscalyear"]
    )
    emissions = emissions.drop( # surplus columns
        columns=[
            "institutionid", "periodenddate", "companyid", "companyname"
        ]
    )

    # finally, write out the gvkeys for linking with security and fundamentals data
    write_ids(emissions, "gvkey", "gvkeys.txt")
    # emissions is preprocessed and ready for modelling
    return emissions


def load_exrts(filename):
    """ load daily currency rates, return monthly rates """

    d_exrts = pd.read_csv(
        filename, 
        parse_dates=["datadate"], 
    )
    d_exrts = d_exrts.sort_values(["curd", "datadate"])
    d_exrts["data_ym"] = d_exrts["datadate"].dt.to_period('M')

    m_exrts = d_exrts.groupby(
        ["curd", "data_ym"]
    ).last(
    ).drop(
        columns="datadate"
    )

    return m_exrts


def load_market_returns(filename):
    mkt_rets = pd.read_csv(
        filename, 
        parse_dates=["datadate"], 
    )
    # year-month index
    mkt_rets["data_ym"] = mkt_rets["datadate"].dt.to_period("M")
    mkt_rets = mkt_rets.set_index("data_ym")
    # compute monthly returns from prices
    mkt_rets["m_mktret"] = mkt_rets["prccm"].pct_change()
    mkt_rets = mkt_rets.dropna()
    # only the final returns are needed
    mkt_rets = mkt_rets[["m_mktret"]]

    return mkt_rets


def load_security_returns(filename, m_exrts):
    security_returns = pd.read_csv(
        filename, 
        parse_dates=["datadate"], 
    )
    security_returns["data_ym"] = security_returns["datadate"].dt.to_period('M')

    # remove missing values of prices, return factors, adjustment factors
    security_returns = security_returns.dropna()

    # index
    security_returns = security_returns.set_index(["gvkey", "iid", "data_ym"])
    security_returns = security_returns.sort_index()

    # prices for returns
    security_returns["adjclose"] = (security_returns["prccm"] / security_returns["ajexm"]) * security_returns["trfm"]

    # join exchange rates to account for the effect of their fluctuations on returns
    security_returns = security_returns.reset_index().merge(
        m_exrts, 
        how="left", 
        left_on=["curcdm", "data_ym"], 
        right_index=True, 
    ).set_index(
        ["gvkey", "iid", "data_ym"]
    )
    # compute raw local and FX returns
    security_returns = pd.concat(
        [
            security_returns, 
            security_returns.groupby(
                level=["gvkey", "iid"]
            )[["adjclose", "exratd_toUSD", "exratd_toGBP"]].pct_change(
            ).rename(
                columns={
                    "adjclose": "local_ret", 
                    "exratd_toUSD": "USD_fxret", 
                    "exratd_toGBP": "GBP_fxret", 
                }
            )
        ], 
        axis=1
    )
    # raw returns w.r.t base currencies
    security_returns["USD_ret"] = (
        (1 + security_returns["local_ret"]) * (1 + security_returns["USD_fxret"]) - 1
    )
    security_returns["GBP_ret"] = (
        (1 + security_returns["local_ret"]) * (1 + security_returns["GBP_fxret"]) - 1
    )

    # durations of returns in months, that can be used to standardise them
    security_returns["ret_mfreq"] = security_returns.reset_index(
    ).groupby(
        ["gvkey", "iid"]
    )["data_ym"].diff(
    ).apply(
        lambda x: x.n if pd.notnull(x) else np.nan
    ).values
    # standardise returns to monthly frequency
    security_returns["m_USD_ret"] = (
        (1 + security_returns["USD_ret"]) ** (1/security_returns["ret_mfreq"])
    ) - 1
    # 
    security_returns["m_GBP_ret"] = (
        (1 + security_returns["GBP_ret"]) ** (1/security_returns["ret_mfreq"])
    ) - 1
    # 
    security_returns["m_local_ret"] = (
        (1 + security_returns["local_ret"]) ** (1/security_returns["ret_mfreq"])
    ) - 1

    # market value
    security_returns["local_mktval"] = security_returns["cshom"] * security_returns["prccm"]
    security_returns["USD_mktval"] = security_returns["local_mktval"] * security_returns["exratd_toUSD"]
    security_returns["GBP_mktval"] = security_returns["local_mktval"] * security_returns["exratd_toGBP"]


def main():
    pass

if __name__ == "__main__":
    main()
