def filter_df(df, col, val):
    """
    Filter for rows with certain values in the desired column in df

    Parameters:
        df : pd.DataFrame
        col : string
        val : value of interest
    """
    return df[df[col] == val]


def filter_dists(dist_mtrx, from_ind, to_ind):
    """
    Filter a distance matrix {dist_mtrx} to keep rows {from_ind} and
    columns {to_ind}

    Parameters:
        dist_mtrx : 2D matrix
        from_ind, to_ind : list of integers
    """

    str_inds = [str(i) for i in to_ind]
    res = dist_mtrx[str_inds].loc[from_ind]

    # reset the indices and columns
    res.reset_index(inplace=True, drop=True)
    cols = [str(i) for i in range(len(to_ind))]
    res.columns = cols

    return res
