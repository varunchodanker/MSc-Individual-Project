import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

WRITE_ON=True

def write_ids(
    df, idname, filename, 
    from_index=False
):
    """ Writes the unique, non-NaN instances of the indicated identifier, within the 
    indicated dataframe, on separate lines of a new file, whose filename should 
    be specified.

    Args:
        df: Dataframe.
        idname: Column name of the identifier with respect to the dataframe.
        filename: The name to use for the newly created file.
    """
    if not WRITE_ON:
        print("WARNING: WRITING ATTEMPTED BUT BLOCKED")
        return

    id_vals = df.index.get_level_values(idname) if from_index else df[idname]
    # 
    with open(filename, "w") as fh:
        for idval in id_vals.dropna().unique():
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

    emissions = read_emissions(emissions_filename)

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
    # remove surplus columns
    emissions = emissions.drop(
        columns=[
            "institutionid", "periodenddate", "companyid", "companyname"
        ]
    )
    # remove rows with missing values for the remaining relevant features
    emissions = emissions.dropna()

    # finalise the dataframe, ready for linking
    emissions = emissions.reset_index(drop=True)
    emissions = emissions.set_index(
        ["gvkey", "fiscalyear"]
    )

    # finally, write out the gvkeys for linking with security and fundamentals data
    write_ids(emissions, "gvkey", "gvkeys.txt", from_index=True)
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
    mkt_rets["m_mktret"] = mkt_rets["prccm"].pct_change() * 100
    mkt_rets = mkt_rets.dropna()
    # only the final returns are needed
    mkt_rets = mkt_rets[["m_mktret"]]

    return mkt_rets


def load_security_returns(
    filename, m_exrts, mkt_rets, 
    drop_outliers=[], keep_cols=None
):
    """ Loads a dataframe with security returns and other associated information 
    like betas. 
    
    Args:
        drop_outliers: list of column names whose outliers should be dropped at 
            finalisation.
        keep_cols: list of columns that will be kept at finalisation, the rest 
            of the columns will be discarded.
    """
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
    # also convert all the returns from fractions to percentages
    security_returns["local_ret"] *= 100
    security_returns["USD_fxret"] *= 100
    security_returns["GBP_fxret"] *= 100
    # 
    security_returns["USD_ret"] *= 100
    security_returns["GBP_ret"] *= 100
    # 
    security_returns["m_USD_ret"] *= 100
    security_returns["m_GBP_ret"] *= 100
    security_returns["m_local_ret"] *= 100

    # market value
    security_returns["local_mktval"] = security_returns["cshom"] * security_returns["prccm"]
    security_returns["USD_mktval"] = security_returns["local_mktval"] * security_returns["exratd_toUSD"]
    security_returns["GBP_mktval"] = security_returns["local_mktval"] * security_returns["exratd_toGBP"]
    # shift one forward to prevent look-ahead bias
    security_returns[
        ["local_mktval", "USD_mktval", "GBP_mktval"]
    ] = security_returns.groupby(
        level=["gvkey", "iid"]
    )[
        ["local_mktval", "USD_mktval", "GBP_mktval"]
    ].shift()

    # compute beta on security return, joining market returns for this purpose
    security_returns = security_returns.join(
        mkt_rets, 
        how="left", 
        on="data_ym", 
    )
    # variances and covariances
    security_return_label = "m_local_ret" # designates the return of interest
    security_betas = security_returns.reset_index(
    ).groupby(
        ["gvkey", "iid"] # within issues
    )[[security_return_label, "m_mktret"]].rolling(
        12
    ).cov().unstack()["m_mktret"].rename(
        columns={
            security_return_label: "cov(i,m)", 
            "m_mktret": "var(m)", 
        }
    ).set_index(security_returns.index)
    # beta only
    security_betas["beta"] = security_betas["cov(i,m)"] / security_betas["var(m)"]
    security_betas = security_betas.drop(columns=["cov(i,m)", "var(m)"])
    # shfit one forward to prevent look-ahead bias
    security_betas = security_betas.groupby(
        level=["gvkey", "iid"]
    ).shift()
    # combine beta back into the main data
    security_returns = pd.concat([security_returns, security_betas], axis=1)

    # finalise the data, dropping due to the missing values from the 
    # shifts, return calculations, etc
    security_returns = security_returns.dropna()
    # removal of rows with outliers is also possible: "m_USD_ret", "beta", "USD_mktval"
    if len(drop_outliers) > 0:
        keep_idx = pd.Series(True, index=security_returns.index)
        for drop_col in drop_outliers:
            keep_idx &= (
                (security_returns[drop_col].quantile(0.001) < security_returns[drop_col])
                & (security_returns[drop_col] < security_returns[drop_col].quantile(0.999))
            )
        security_returns = security_returns[keep_idx]
    # Prepare for linking based on the prior year - for fundamentals and emissions
    security_returns["datayear"] = security_returns.reset_index()["data_ym"].dt.year.values
    security_returns["datayear-1"] = security_returns["datayear"] - 1
    # filtering down of the columns, to those that are relevant, is also possible
    if keep_cols is not None:
        security_returns = security_returns[keep_cols]

    return security_returns


def load_fundamentals(filename, m_exrts, keep_cols=None):
    """ 
    
    Args:

    Returns:
        loaded dataframe with the relevant company fundamentals information
    """
    fundamentals = pd.read_csv(
        filename, 
        parse_dates=["datadate"]
    )
    fundamentals = fundamentals.rename(columns={"fyear": "fiscalyear"})
    # consolidate double-reporting, per fiscal year, by companies (occurs due 
    # to different reporting formats)
    fundamentals = fundamentals.groupby(
        ["gvkey", "fiscalyear"]
    ).first()
    # drop rows if there are still any null values left despite consolidation
    fundamentals = fundamentals.dropna()

    # standardise values to USD, processing exchange rates so that they are 
    # appropriate for this purpose, and then joining them so that they can 
    # be applied.
    m_exrts_flat = m_exrts.reset_index()
    m_exrts_flat["datayear"] = m_exrts_flat["data_ym"].dt.year
    # yearly balance sheet figures conversion rates
    y_bs_exrts = m_exrts_flat.drop(columns="data_ym").groupby(
        ["curd", "datayear"]
    ).last()[["exratd_toUSD"]].rename(
        columns={"exratd_toUSD": "bs_toUSD"}
    )
    # yearly income statement figures conversion rates
    y_is_exrts = m_exrts_flat.drop(columns="data_ym").groupby(
        ["curd", "datayear"]
    ).mean()[["exratd_toUSD"]].rename(columns={"exratd_toUSD": "is_toUSD"})
    # combine the above two sets of rates
    y_fs_exrts = pd.concat(
        [y_bs_exrts, y_is_exrts], 
        axis=1
    )
    # join them to the main table
    fundamentals = fundamentals.reset_index().merge(
        y_fs_exrts, 
        how="left", 
        left_on=["curcd", "fiscalyear"], 
        right_index=True, 
    ).set_index(
        ["gvkey", "fiscalyear"]
    )
    # apply the rates
    fundamentals["at"] *= fundamentals["bs_toUSD"]
    fundamentals["ceq"] *= fundamentals["bs_toUSD"]
    # 
    fundamentals["oiadp"] *= fundamentals["is_toUSD"]
    fundamentals["revt"] *= fundamentals["is_toUSD"]

    # determine investment in each year by percentage change in assets
    fundamentals["investment"] = fundamentals.groupby(
        level="gvkey"
    )["at"].pct_change()
    # null for the first fiscalyear entry for companies

    # determine operating profitability by operating profit margin
    fundamentals["opm"] = fundamentals["oiadp"] / fundamentals["revt"]
    # some inf values can be caused here due to div by 0

    # finalise the data, dropping all rows with null or non-finite values
    with pd.option_context('mode.use_inf_as_null', True):
        fundamentals = fundamentals.dropna()
    # filtering down of the columns is also possible
    if keep_cols is not None:
        fundamentals = fundamentals[keep_cols]

    return fundamentals


def main():
    m_exrts = load_exrts("phase1_exrts.csv")
    mkt_rets = load_market_returns("phase1_index.csv")

    write_emission_companyids("phase1_emissions.csv")
    emissions = load_emissions("phase1_emissions.csv", "cid_gvkey_map.csv")
    print(emissions)
    fundamentals = load_fundamentals("phase1_fundamentals.csv", m_exrts)
    # security_returns = load_security_returns("phase1_returns.csv", m_exrts, mkt_rets)

if __name__ == "__main__":
    main()
