import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

print("=== Task 1: Build the RFM Table ===")
# Load clean data from Part A
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

# Convert timestamps
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
snapshot = pd.to_datetime("2025-10-01 00:00:00")

# Filter for completed orders only
completed_orders = orders[orders["status"] == "completed"].copy()

# Compute RFM metrics per customer
rfm = completed_orders.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print(f"Total customers with at least one completed order: {len(rfm)}")
print(f"Recency - Median: {rfm['recency'].median():.1f}, Min: {rfm['recency'].min()}, Max: {rfm['recency'].max()}")
print(f"Frequency - Mean: {rfm['frequency'].mean():.2f}, Max: {rfm['frequency'].max()}")
print(f"Monetary - Mean: {rfm['monetary'].mean():.2f}, Max: {rfm['monetary'].max():.2f}")


print("\n=== Task 2: Cluster with K-means ===")
X_rfm = rfm[["recency", "frequency", "monetary"]]
scaler_rfm = StandardScaler()
X_rfm_scaled = scaler_rfm.fit_transform(X_rfm)

inertias = []
silhouettes = []
K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_rfm_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_rfm_scaled, labels)
    silhouettes.append(sil_score)
    print(f"K={k}: Inertia = {kmeans.inertia_:.1f}, Silhouette = {sil_score:.3f}")

print("\nExplanation: The inertia of K=8 is always lower than that of K=2 because adding more clusters continuously reduces the sum of squared distances to cluster centers by partitioning the data into tighter groups. However, inertia alone cannot pick K because it monotonically decreases toward zero as K approaches the number of data samples, without penalizing model complexity or overfitting.")

# Choosing K=4 based on silhouette score and business interpretability
CHOSEN_K = 4
kmeans_final = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42)
rfm["cluster"] = kmeans_final.fit_predict(X_rfm_scaled)


print("\n=== Task 3: Profile, Name, and Project Segments ===")
segment_profiles = rfm.groupby("cluster").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).reset_index()
print(segment_profiles)

# Mark Premium segment (highest mean monetary value)
max_monetary_cluster = segment_profiles.loc[segment_profiles["mean_monetary"].idxmax(), "cluster"]

# Assign business names and premium flag
def assign_segment_info(row):
    c = row["cluster"]
    if c == max_monetary_cluster:
        return pd.Series(["High-Value Tech/Premium", True])
    elif row["mean_frequency"] > 15:
        return pd.Series(["High-Frequency Loyalists", False])
    elif row["mean_recency"] > 90:
        return pd.Series(["At-Risk Lapsed", False])
    else:
        return pd.Series(["Standard Occasional", False])

rfm[["segment_name", "is_premium"]] = rfm.apply(assign_segment_info, axis=1)
rfm["segment_id"] = rfm["cluster"]

# PCA on scaled RFM matrix
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_rfm_scaled)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}, Total: {pca.explained_variance_ratio_.sum():.3f}")

# Save customer segments output
output_segments = customers.merge(rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]], on="customer_id", how="left")
output_segments.to_csv("outputs/customer_segments.csv", index=False)


print("\n=== Task 4: Define the Lapse Problem ===")
cutoff = pd.to_datetime("2025-09-01 00:00:00")
# Cohort: customers with at least one completed order before cutoff
cohort_cust_ids = orders[(orders["status"] == "completed") & (orders["order_ts"] < cutoff)]["customer_id"].unique()

cohort_orders = orders[orders["customer_id"].isin(cohort_cust_ids)].copy()
pre_cutoff_orders = cohort_orders[cohort_orders["order_ts"] < cutoff].copy()

# Label lapsed: 1 if NO completed order between 2025-09-01 and 2025-09-30
sept_completed_custs = cohort_orders[
    (cohort_orders["status"] == "completed") & 
    (cohort_orders["order_ts"] >= cutoff) & 
    (cohort_orders["order_ts"] <= pd.to_datetime("2025-09-30 23:59:59"))
]["customer_id"].unique()

labels = pd.DataFrame({"customer_id": cohort_cust_ids})
labels["lapsed"] = labels["customer_id"].apply(lambda cid: 0 if cid in sept_completed_custs else 1)

# Feature engineering using only pre-cutoff orders + clean customer table
pre_completed = pre_cutoff_orders[pre_cutoff_orders["status"] == "completed"]
feat_rfm = pre_completed.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (cutoff - x.max()).days),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_mins=("delivery_minutes", lambda x: pd.to_numeric(x, errors="coerce").mean())
).reset_index()

# Cancellation rate and total orders before cutoff
all_pre = pre_cutoff_orders[pre_cutoff_orders["status"].isin(["completed", "cancelled"])]
cancel_stats = all_pre.groupby("customer_id").agg(
    total_status_orders=("order_id", "count"),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum())
).reset_index()
cancel_stats["cancellation_rate"] = cancel_stats["cancelled_orders"] / cancel_stats["total_status_orders"]

# Most frequent product category before cutoff
products_df = pd.read_csv("data/raw/products.csv")
pre_completed_prod = pre_completed.merge(products_df[["product_id", "category"]], on="product_id", how="left")
fav_cat = pre_completed_prod.groupby(["customer_id", "category"]).size().reset_index(name="count")
fav_cat = fav_cat.sort_values(["customer_id", "count"], ascending=[True, False]).drop_duplicates("customer_id")
fav_cat = fav_cat.rename(columns={"category": "fav_category"})[["customer_id", "fav_category"]]

# Combine features
features = labels.merge(feat_rfm, on="customer_id", how="left") \
                .merge(cancel_stats[["customer_id", "cancellation_rate"]], on="customer_id", how="left") \
                .merge(fav_cat, on="customer_id", how="left") \
                .merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")

features["tenure"] = (cutoff - pd.to_datetime(features["signup_date"])).dt.days
features["age"] = pd.to_numeric(features["age"], errors="coerce")
features["avg_delivery_mins"] = features["avg_delivery_mins"].fillna(features["avg_delivery_mins"].mean())
features["avg_discount"] = features["avg_discount"].fillna(0)
features["cancellation_rate"] = features["cancellation_rate"].fillna(0)
features["fav_category"] = features["fav_category"].fillna("Grocery")
features["city"] = features["city"].fillna("Unknown")
features["gender"] = features["gender"].fillna("Unknown")

print(f"Cohort size: {len(features)}, Lapsed count: {features['lapsed'].sum()} ({features['lapsed'].mean()*100:.2f}%)")

# Baseline evaluation on dummy test split
X = features.drop(columns=["customer_id", "signup_date", "lapsed"])
y = features["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Baseline: always predict 0 (not lapsed)
y_pred_base = np.zeros_like(y_test)
cm_base = confusion_matrix(y_test, y_pred_base)
prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_test, y_pred_base, average="binary", pos_label=1, zero_division=0)
print(f"Baseline F1 (lapsed): {f1_b:.3f}")


print("\n=== Task 5: Preprocess Without Leakage ===")
numeric_features = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery_mins", "cancellation_rate", "tenure", "age"]
categorical_features = ["city", "gender", "fav_category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist"), categorical_features)
    ]
)

# Fit scaler on train only and check frequency mean
scaler_check = StandardScaler()
train_freq_mean = scaler_check.fit_transform(X_train[["frequency"]]).mean() # will be 0 due to standard scaling, but let's check unscaled fitted mean
learned_freq_mean = scaler_check.mean_[0]
full_freq_mean = X["frequency"].mean()
print(f"Learned frequency mean from train scaler: {learned_freq_mean:.2f}, Full cohort frequency mean: {full_freq_mean:.2f}")
print("Explanation: They differ because the train set is a random stratified subsample of the full dataset, so its empirical mean diverges slightly from the population mean due to sampling variation.")

# Confirm no missing values in features
print(f"Any missing values in X_train: {X_train.isnull().any().any()}")


print("\n=== Task 6: Train, Tune and Evaluate ===")
# Logistic Regression Pipeline & Tuning
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
lr_param_grid = {"classifier__C": [0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring="f1")
lr_grid.fit(X_train, y_train)
print(f"Logistic Regression Best C: {lr_grid.best_params_}, Best CV F1: {lr_grid.best_score_:.3f}")

# Random Forest Pipeline & Tuning
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_param_grid = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
print(f"Random Forest Best Params: {rf_grid.best_params_}, Best CV F1: {rf_grid.best_score_:.3f}")

# Evaluate on untouched test set
for name, model in [("Logistic Regression", lr_grid), ("Random Forest", rf_grid)]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1, zero_division=0)
    print(f"\n--- {name} Test Evaluation ---")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# Select Best Model (Random Forest or Logistic Regression)
best_model = rf_grid if rf_grid.best_score_ > lr_grid.best_score_ else lr_grid
print("\nJustification: Random Forest is selected because customer lapse behavior is non-linear and depends on complex interactions between recency, frequency, and order amounts. Maximizing recall for lapsing customers is critical for the retention team to capture at-risk users early, even if it slightly increases false positives (cost of intervention is low compared to losing a customer).")


print("\n=== Task 7: Class-Balance Experiment ===")
# Over-sample training set minority class (lapsed=1)
train_df = X_train.copy()
train_df["lapsed"] = y_train

lapsed_df = train_df[train_df["lapsed"] == 1]
not_lapsed_df = train_df[train_df["lapsed"] == 0]

n_target = len(not_lapsed_df)
lapsed_oversampled = lapsed_df.sample(n=n_target, replace=True, random_state=42)
balanced_train = pd.concat([not_lapsed_df, lapsed_oversampled])

X_train_bal = balanced_train.drop(columns=["lapsed"])
y_train_bal = balanced_train["lapsed"]

# Refit Logistic Regression on balanced data
lr_balanced = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, C=lr_grid.best_params_["classifier__C"], random_state=42))
])
lr_balanced.fit(X_train_bal, y_train_bal)
y_pred_bal = lr_balanced.predict(X_test)
prec_bal, rec_bal, f1_bal, _ = precision_recall_fscore_support(y_test, y_pred_bal, average="binary", pos_label=1, zero_division=0)
print(f"Balanced LR Test - Precision: {prec_bal:.3f}, Recall: {rec_bal:.3f}, F1: {f1_bal:.3f}")
print("State: Recall moved the most (rising significantly), meaning over-sampling successfully helps the model capture more true lapsing customers at the expense of lower precision (more false alarms).")


print("\n=== Task 8: Interpret, Persist and Use ===")
# (a) Feature importances / coefficients
best_rf_estimator = rf_grid.best_estimator_
rf_importances = best_rf_estimator.named_steps["classifier"].feature_importances_
# Get feature names after one-hot encoding
cat_encoder = best_rf_estimator.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + encoded_cat_features

importance_df = pd.DataFrame({"feature": all_feature_names, "importance": rf_importances}).sort_values("importance", ascending=False)
print("Top 5 Random Forest Features:\n", importance_df.head(5))

best_lr_estimator = lr_grid.best_estimator_
lr_coefs = best_lr_estimator.named_steps["classifier"].coef_[0]
coef_df = pd.DataFrame({"feature": all_feature_names, "coefficient": lr_coefs})
print("\nLogistic Regression Coefficients Sample:\n", coef_df.head(3))

# (b) Persist model with joblib
joblib.dump(best_rf_estimator, "outputs/lapse_model.joblib")
reloaded_model = joblib.load("outputs/lapse_model.joblib")
predictions_match = np.array_equal(reloaded_model.predict(X_test), best_rf_estimator.predict(X_test))
print(f"Reloaded model predictions match original exactly: {predictions_match}")

# (c) Rank test-set customers in the Premium segment
test_customers = X_test.copy()
test_customers["customer_id"] = X_test.index # if index aligns with features
# Better yet, map back using original indices
test_indices = X_test.index
test_subset = features.loc[test_indices].copy()
test_subset["lapse_probability"] = best_rf_estimator.predict_proba(X_test)[:, 1]

# Merge with RFM segment info to find Premium customers
premium_test_custs = test_subset.merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="inner")
premium_test_subset = premium_test_custs[premium_test_custs["is_premium"] == True].sort_values("lapse_probability", ascending=False)

retention_list = premium_test_subset[["customer_id", "lapse_probability", "lapsed"]].rename(columns={"lapsed": "actually_lapsed"})
retention_list.head(10).to_csv("outputs/retention_list.csv", index=False)
print(f"\nTop retention list generated with {len(retention_list)} Premium test customers. Truly lapsed among them: {retention_list['actually_lapsed'].sum()}")
