import tabulate
# Import prepare graph data

# create and nicely print a dataframe with columns for model names and then show the differences between 1. the average values for the control and coordinate(none)
# and 2. the average values for control and coordinate(COT)

def print_nice_dataframe(df, max_rows=20, show_index=False):
    """Generic function for nicely printing any DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        max_rows (int): Maximum number of rows to display
        show_index (bool): Whether to show the index in the output
    """
    if len(df) > max_rows:
        print(f"Displaying first {max_rows} rows (total: {len(df)}):\n")
        print(tabulate(df.head(max_rows), headers='keys', 
                     tablefmt='grid', showindex=show_index))
    else:
        print(tabulate(df, headers='keys', tablefmt='grid', 
                     showindex=show_index))
