import pandas as pd


class GroupEstimate:
    """Estimate a value for each group of categorical observations."""

    def __init__(self, estimate):
        """Initialize the model with a mean or median estimate."""
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        
        self.estimate = estimate


    def fit(self, X, y):
        """Calculate and save an estimate for each category group"""
        data = X.copy()
        data["_y"] = y

        grouped = data.groupby(list(X.columns), observed=True)["_y"]

        if self.estimate == "mean":
            self.group_estimates = grouped.mean()
        else:
            self.group_estimates = grouped.median()

        self.feature_names = list(X.columns)
        return None

    def predict(self, X):
        """Predict values by looking up each row's category group."""
        X_ = pd.DataFrame(X, columns=self.feature_names)
        estimates = self.group_estimates.rename("_estimate").reset_index()

        predictions = X_.merge(
            estimates, on=self.feature_names, how="left"
        )["_estimate"]

        missing_count = predictions.isna().sum()

        if missing_count > 0:
            print(f"{missing_count} observations had no matching group.")

        return predictions.to_numpy()
    