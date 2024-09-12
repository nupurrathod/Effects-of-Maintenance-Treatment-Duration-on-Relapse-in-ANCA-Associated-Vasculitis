# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import expit

# %% [markdown]
# # Data Generation

# %%
def generate_synthetic_data(n_samples=1000):
    X = np.random.randn(n_samples, 11)  # X1 to X11 from N(0,1)
    X = np.hstack((X, np.random.binomial(1, 0.5, (n_samples, 9))))  # X12 to X20 from Bernoulli(0.5)

    F_X = -2 + 0.028*X[:,0] - 0.374*X[:,1] - 0.03*X[:,2] + 0.118*X[:,3] - 0.394*X[:,10] + 0.875*X[:,11] + 0.9*X[:,12]
    prob_T = expit(F_X)
    T = np.random.binomial(1, prob_T)

    Y = []
    true_effect = []
    for model in range(3):
        if model == 0:
            Y_model = 2.455 - (1-T) * (0.4*X[:,0] + 0.154*X[:,1] - 0.152*X[:,10] - 0.126*X[:,11]) - T * (0.254*X[:,1] - 0.152*X[:,10] - 0.4*X[:,1]**2 - 0.126*X[:,11] > 0)
        elif model == 1:
            Y_model = 2.455 - (1-T) * np.sin(0.4*X[:,0] + 0.154*X[:,1] - 0.152*X[:,10] - 0.126*X[:,11]) - T * (0.254*X[:,1] - 0.152*X[:,10] - 0.4*X[:,1]**2 - 0.126*X[:,11] > 0)
        else:
            Y_model = 2.455 - (1-T) * np.sin(0.4*X[:,0] + 0.154*X[:,1] - 0.152*X[:,10] - 0.126*X[:,11]) - T * (0.254*X[:,1] - 0.152*X[:,10] - 0.126*X[:,2] - 0.4*X[:,3] - 0.4*X[:,4]**2 > 0)

        Y_model += np.random.normal(0, 0.1, n_samples)  # Add noise with σ = 0.1
        Y.append(Y_model)

        Y_t1 = Y_model.copy()
        Y_t0 = Y_model.copy()
        Y_t1[T == 0] = 2.455 - (0.254*X[T==0,1] - 0.152*X[T==0,10] - 0.4*X[T==0,1]**2 - 0.126*X[T==0,11] > 0)
        Y_t0[T == 1] = 2.455 - (0.4*X[T==1,0] + 0.154*X[T==1,1] - 0.152*X[T==1,10] - 0.126*X[T==1,11])
        true_effect.append(Y_t1 - Y_t0)

    return X, T, Y, prob_T, true_effect

# %% [markdown]
# # Model Implementations

# %%
class VirtualTwins:
    def __init__(self, include_interactions=True):
        self.include_interactions = include_interactions
        self.rf = RandomForestRegressor(n_estimators=1000, max_features=7, min_samples_leaf=3, random_state=42, oob_score=True)

    def fit(self, X, T, y):
        if self.include_interactions:
            X_with_interactions = np.column_stack((X, T, X * T[:, np.newaxis]))
        else:
            X_with_interactions = np.column_stack((X, T))
        self.rf.fit(X_with_interactions, y)
        print(f"Virtual Twins OOB Score: {self.rf.oob_score_}")

    def predict_ite(self, X):
        if self.include_interactions:
            X_t1 = np.column_stack((X, np.ones(X.shape[0]), X))
            X_t0 = np.column_stack((X, np.zeros(X.shape[0]), np.zeros_like(X)))
        else:
            X_t1 = np.column_stack((X, np.ones(X.shape[0])))
            X_t0 = np.column_stack((X, np.zeros(X.shape[0])))

        y_t1 = self.rf.predict(X_t1)
        y_t0 = self.rf.predict(X_t0)

        return y_t1 - y_t0

# %%
class CounterfactualRF:
    def __init__(self):
        self.rf_t1 = RandomForestRegressor(n_estimators=1000, max_features=7, min_samples_leaf=3, random_state=42, oob_score=True )
        self.rf_t0 = RandomForestRegressor(n_estimators=1000, max_features=7, min_samples_leaf=3, random_state=42, oob_score=True )

    def fit(self, X, T, y):
        self.rf_t1.fit(X[T == 1], y[T == 1])
        print(f"Counterfactual RF (T=1) OOB Score: {self.rf_t1.oob_score_}")
        
        self.rf_t0.fit(X[T == 0], y[T == 0])
        print(f"Counterfactual RF (T=0) OOB Score: {self.rf_t0.oob_score_}")

    def predict_ite(self, X):
        y_t1 = self.rf_t1.predict(X)
        y_t0 = self.rf_t0.predict(X)
        return y_t1 - y_t0

# %%
class SyntheticForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.base_learners = []
        self.synthetic_forest = None

    def fit(self, X, y):
        param_grid = {
            'n_estimators': [100],
            'max_features': [1, 'sqrt', 'log2'],
            'min_samples_leaf': [1, 3, 5, 10]
        }

        for params in ParameterGrid(param_grid):
            rf = RandomForestRegressor(**params, oob_score=True, random_state=42)
            rf.fit(X, y)
            self.base_learners.append(rf)

        synthetic_features = np.column_stack([rf.oob_prediction_ for rf in self.base_learners])
        X_combined = np.column_stack([X, synthetic_features])

        self.synthetic_forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42, 
            oob_score=True 
        )
        self.synthetic_forest.fit(X_combined, y)
        print(f"SyntheticForest OOB Score: {self.synthetic_forest.oob_score_}")

        return self

    def predict(self, X):
        synthetic_features = np.column_stack([rf.predict(X) for rf in self.base_learners])
        X_combined = np.column_stack([X, synthetic_features])
        return self.synthetic_forest.predict(X_combined)

# %%
class CounterfactualSyntheticRF:
    def __init__(self):
        self.sf_t1 = SyntheticForest()
        self.sf_t0 = SyntheticForest()

    def fit(self, X, T, y):
        self.sf_t1.fit(X[T == 1], y[T == 1])
        self.sf_t0.fit(X[T == 0], y[T == 0])

    def predict_ite(self, X):
        y_t1 = self.sf_t1.predict(X)
        y_t0 = self.sf_t0.predict(X)
        return y_t1 - y_t0

# %% [markdown]
# # Simulation with Synthetic Data

# %%
def evaluate_ite_estimation(ite_pred, true_effect, propensity_scores, n_strata=100):
    strata = pd.qcut(propensity_scores, n_strata, labels=False)

    bias = []
    rmse = []

    for stratum in range(n_strata):
        mask = strata == stratum
        bias_stratum = np.mean(ite_pred[mask] - true_effect[mask])
        rmse_stratum = np.sqrt(mean_squared_error(true_effect[mask], ite_pred[mask]))

        bias.append(bias_stratum)
        rmse.append(rmse_stratum)

    return np.array(bias), np.array(rmse)

# %%
def run_simulation(n_samples, n_reps):
    results = []
    for model in range(3):
        bias_vt, rmse_vt = [], []
        bias_cf, rmse_cf = [], []
        bias_syncf, rmse_syncf = [], []

        for _ in range(n_reps):
            X, T, Y, prob_T, true_effect = generate_synthetic_data(n_samples)

            vt = VirtualTwins(include_interactions=True)
            vt.fit(X, T, Y[model])
            ite_vt = vt.predict_ite(X)

            cf_rf = CounterfactualRF()
            cf_rf.fit(X, T, Y[model])
            ite_cf = cf_rf.predict_ite(X)

            syncf = CounterfactualSyntheticRF()
            syncf.fit(X, T, Y[model])
            ite_syncf = syncf.predict_ite(X)

            bias_vt_rep, rmse_vt_rep = evaluate_ite_estimation(ite_vt, true_effect[model], prob_T)
            bias_cf_rep, rmse_cf_rep = evaluate_ite_estimation(ite_cf, true_effect[model], prob_T)
            bias_syncf_rep, rmse_syncf_rep = evaluate_ite_estimation(ite_syncf, true_effect[model], prob_T)

            bias_vt.append(bias_vt_rep)
            rmse_vt.append(rmse_vt_rep)
            bias_cf.append(bias_cf_rep)
            rmse_cf.append(rmse_cf_rep)
            bias_syncf.append(bias_syncf_rep)
            rmse_syncf.append(rmse_syncf_rep)

        results.append({
            'model': model + 1,
            'vt_bias': np.mean(bias_vt),
            'vt_rmse': np.mean(rmse_vt),
            'cf_bias': np.mean(bias_cf),
            'cf_rmse': np.mean(rmse_cf),
            'syncf_bias': np.mean(bias_syncf),
            'syncf_rmse': np.mean(rmse_syncf)
        })

    return pd.DataFrame(results)

# %% [markdown]
# ## Run simulations
# 

# %%
print("Running simulations...")
results_500 = run_simulation(n_samples=500, n_reps=10)

print("\nResults for n=500 rep=10:")
print(results_500)

# %%
print("Running simulations...")
results_500_100 = run_simulation(n_samples=500, n_reps=100)

print("\nResults for n=500 rep=100:")
print(results_500_100)

# %%
results_5000 = run_simulation(n_samples=5000, n_reps=250)

print("\nResults for n=5000:")
print(results_5000)

# %%
def plot_simulation_results(results, n_samples):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for i, metric in enumerate(['bias', 'rmse']):
        ax = axes[i]

        x = np.arange(3)
        width = 0.25

        ax.bar(x - width, results[f'vt_{metric}'], width, label='Virtual Twins')
        ax.bar(x, results[f'cf_{metric}'], width, label='Counterfactual RF')
        ax.bar(x + width, results[f'syncf_{metric}'], width, label='Synthetic CF')

        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} for n={n_samples}')
        ax.set_xticks(x)
        ax.set_xticklabels(['Model 1', 'Model 2', 'Model 3'])
        ax.legend()

    plt.suptitle(f'Simulation Results for n={n_samples}', fontsize=16)
    plt.tight_layout()
    plt.show()

# %%
# Visualize simulation results for n=500 and 10 reps
plot_simulation_results(results_500, 500)

# %%
# Visualize simulation results for n=500 and 100 reps
plot_simulation_results(results_500_100, 500)

# %%
# Visualize simulation results for n=5000 and 250 reps
plot_simulation_results(results_5000, 5000)

# %% [markdown]
# # Analysis on Real world Data

# %%
df_16 = pd.read_csv('filtered_16.csv')
df_20 = pd.read_csv('filtered_20.csv')
df_24 = pd.read_csv('filtered_24.csv')

# %% [markdown]
# ## Data Pre-processing

# %%
features = [
    'Corticosteroids', 'Gender', 'Ethnicity', 'Smoking', 'Any.Induction.Treatment',
    'Any.Maintenance.Treatment', 'Maintenance.treatment.received..choice.Oral.corticosteroids.',
    'Maintenance.treatment.received..choice.Rituximab.',
    'Maintenance.treatment.received..choice.Azathioprine.',
    'Maintenance.treatment.received..choice.Cyclophosphamide.',
    'Maintenance.treatment.received..choice.MMF.',
    'Maintenance.treatment.received..choice.Methotrexate.',
    'Maintenance.treatment.received..choice.Avacopan..C5aR.inhibitor..',
    'End.stage.kidney.disease',
]
treatment_16 = 'treatment_on_16_plus'
treatment_20 = 'treatment_on_20_plus'
treatment_24 = 'treatment_on_24_plus'
target = 'relapse'

# %%
# Handle missing values

df_16.fillna('Unknown', inplace=True)
df_20.fillna('Unknown', inplace=True)
df_24.fillna('Unknown', inplace=True)

# Encode categorical variables
df_16_encoded = pd.get_dummies(df_16[features + [treatment_16, target]], drop_first=True)
df_20_encoded = pd.get_dummies(df_20[features + [treatment_20, target]], drop_first=True)
df_24_encoded = pd.get_dummies(df_24[features + [treatment_24, target]], drop_first=True)

# %% [markdown]
# ## Exploratory data analysis (EDA)

# %%
df_16.describe()

# %%
df_20.describe()

# %%
df_24.describe()

# %%
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap for 16 Months Data')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df[target], kde=True)
plt.title('Distribution of Relapse')

# %% [markdown]
# ## Model training

# %%
def split_data(x = 24):
    if x == 24:
        X = np.array(df_24_encoded.drop([treatment_24, target], axis=1))
        T = np.array(df_24_encoded[treatment_24])
        y = np.array(df_24_encoded[target])
    elif x == 20:
        X = np.array(df_20_encoded.drop([treatment_20, target], axis=1))
        T = np.array(df_20_encoded[treatment_20])
        y = np.array(df_20_encoded[target])
    elif x == 16:
        X = np.array(df_16_encoded.drop([treatment_16, target], axis=1))
        T = np.array(df_16_encoded[treatment_16])
        y = np.array(df_16_encoded[target])
    else: 
        print('Enter correct value')

    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.2, random_state=42)   
    return (X_train, X_test, T_train, T_test, y_train, y_test)

# %%
def calculate_ite_rmse(ite_pred, y, T):
    """
    Calculate RMSE of ITE predictions.
    
    This function calculates an approximation of the RMSE for ITE predictions
    by comparing the difference in outcomes between treated and control groups
    to the predicted ITE.
    """
    y_treated = y[T == 1]
    y_control = y[T == 0]
    observed_effect = np.mean(y_treated) - np.mean(y_control)
    
    rmse = np.sqrt(mean_squared_error([observed_effect] * len(ite_pred), ite_pred))
    
    return rmse

# %%
def Train_models(t = 24):
    (X_train, X_test, T_train, T_test, y_train, y_test) = split_data(t)

    vt = VirtualTwins(include_interactions=True)
    vt.fit(X_train, T_train, y_train)
    ite_vt = vt.predict_ite(X_test)

    cf_rf = CounterfactualRF()
    cf_rf.fit(X_train, T_train, y_train)
    ite_cf = cf_rf.predict_ite(X_test)

    syncf = CounterfactualSyntheticRF()
    syncf.fit(X_train, T_train, y_train)
    ite_syncf = syncf.predict_ite(X_test)

    rmse_vt = calculate_ite_rmse(ite_vt, y_test, T_test)
    rmse_cf = calculate_ite_rmse(ite_cf, y_test, T_test)
    rmse_syncf = calculate_ite_rmse(ite_syncf, y_test, T_test)

    return (ite_vt, ite_cf, ite_syncf, rmse_vt, rmse_cf, rmse_syncf)

# %%
results_16 = Train_models(16)
results_20 = Train_models(20)
results_24 = Train_models(24)

# %%
ite_vt_16, ite_cf_16, ite_syncf_16, rmse_vt_16, rmse_cf_16, rmse_syncf_16 = results_16
ite_vt_20, ite_cf_20, ite_syncf_20, rmse_vt_20, rmse_cf_20, rmse_syncf_20 = results_20
ite_vt_24, ite_cf_24, ite_syncf_24, rmse_vt_24, rmse_cf_24, rmse_syncf_24 = results_24

# %% [markdown]
# ## Evaluation

# %%
def compare_ite_statistics(results_16, results_20, results_24):
    ite_vt_16, ite_cf_16, ite_syncf_16 = results_16[:3]
    ite_vt_20, ite_cf_20, ite_syncf_20 = results_20[:3]
    ite_vt_24, ite_cf_24, ite_syncf_24 = results_24[:3]

    stats = {
        '16 months': {
            'Virtual Twins': ite_vt_16,
            'Counterfactual RF': ite_cf_16,
            'Counterfactual Synthetic RF': ite_syncf_16
        },
        '20 months': {
            'Virtual Twins': ite_vt_20,
            'Counterfactual RF': ite_cf_20,
            'Counterfactual Synthetic RF': ite_syncf_20
        },
        '24 months': {
            'Virtual Twins': ite_vt_24,
            'Counterfactual RF': ite_cf_24,
            'Counterfactual Synthetic RF': ite_syncf_24
        }
    }

    for timeframe in stats:
        for model in stats[timeframe]:
            ite = stats[timeframe][model]
            stats[timeframe][model] = {
                'Mean': np.mean(ite),
                'Median': np.median(ite),
                'Std Dev': np.std(ite),
                'Min': np.min(ite),
                'Max': np.max(ite),
                '25th Percentile': np.percentile(ite, 25),
                '75th Percentile': np.percentile(ite, 75)
            }

    df_stats = pd.DataFrame.from_dict({(i,j): stats[i][j] 
                                       for i in stats.keys() 
                                       for j in stats[i].keys()},
                                      orient='index')

    print("ITE Statistics Comparison:")
    print(df_stats.to_string())

    print("\nComparative Insights:")
    for metric in ['Mean', 'Median', 'Std Dev']:
        print(f"\n{metric} ITE Comparison:")
        for timeframe in stats:
            print(f"  {timeframe}:")
            values = [stats[timeframe][model][metric] for model in stats[timeframe]]
            best_model = list(stats[timeframe].keys())[np.argmax(values) if metric != 'Std Dev' else np.argmin(values)]
            print(f"    Best model: {best_model} ({max(values) if metric != 'Std Dev' else min(values):.4f})")

    consistency_scores = {model: np.std([stats[timeframe][model]['Mean'] for timeframe in stats]) 
                          for model in stats['16 months']}
    most_consistent_model = min(consistency_scores, key=consistency_scores.get)
    print(f"\nMost consistent model across timeframes: {most_consistent_model}")
    print(f"Consistency score (lower is better): {consistency_scores[most_consistent_model]:.4f}")

compare_ite_statistics(results_16, results_20, results_24)

# %%
def plot_rmse_comparison():
    models = ['Virtual Twins', 'Counterfactual RF', 'Counterfactual Synthetic RF']
    rmse_16 = [rmse_vt_16, rmse_cf_16, rmse_syncf_16]
    rmse_20 = [rmse_vt_20, rmse_cf_20, rmse_syncf_20]
    rmse_24 = [rmse_vt_24, rmse_cf_24, rmse_syncf_24]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, rmse_16, width, label='16 months', alpha=0.7)
    rects2 = ax.bar(x, rmse_20, width, label='20 months', alpha=0.7)
    rects3 = ax.bar(x + width, rmse_24, width, label='24 months', alpha=0.7)

    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison Across Models and Timeframes')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_rmse_comparison()

# %%
def plot_relapse_rate_by_timeframe():
    timeframes = [16, 20, 24]
    
    relapse_rate_16 = df_16[target].mean()
    relapse_rate_20 = df_20[target].mean()
    relapse_rate_24 = df_24[target].mean()
    
    relapse_rates = [relapse_rate_16, relapse_rate_20, relapse_rate_24]
    
    plt.figure(figsize=(10, 6))
    plt.plot(timeframes, relapse_rates, marker='o', linestyle='-', color='b', label='Relapse Rate')
    plt.xlabel('Timeframe (months)')
    plt.ylabel('Average Relapse Rate')
    plt.title('Relapse Rate by Timeframe')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

plot_relapse_rate_by_timeframe()

# %%
def plot_ite_vs_relapse_rate():
    timeframes = [16, 20, 24]
    
    mean_ite_vt = [np.mean(ite_vt_16), np.mean(ite_vt_20), np.mean(ite_vt_24)]
    mean_ite_cf = [np.mean(ite_cf_16), np.mean(ite_cf_20), np.mean(ite_cf_24)]
    mean_ite_syncf = [np.mean(ite_syncf_16), np.mean(ite_syncf_20), np.mean(ite_syncf_24)]

    relapse_rate_16 = df_16[target].mean()
    relapse_rate_20 = df_20[target].mean()
    relapse_rate_24 = df_24[target].mean()
    relapse_rates = [relapse_rate_16, relapse_rate_20, relapse_rate_24]
    
    plt.figure(figsize=(12, 8))
    plt.plot(timeframes, mean_ite_syncf, marker='^', linestyle='-', color='b', label='Counterfactual Synthetic RF ITE')
    
    plt.bar(timeframes, relapse_rates, width=2, alpha=0.3, color='grey', label='Relapse Rate')
    
    plt.xlabel('Timeframe (months)')
    plt.ylabel('Value')
    plt.title('ITE vs. Relapse Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

plot_ite_vs_relapse_rate()

# %%
mean_ite_vt = [np.mean(ite_vt_16), np.mean(ite_vt_20), np.mean(ite_vt_24)]
mean_ite_cf = [np.mean(ite_cf_16), np.mean(ite_cf_20), np.mean(ite_cf_24)]
mean_ite_syncf = [np.mean(ite_syncf_16), np.mean(ite_syncf_20), np.mean(ite_syncf_24)]

relapse_rate_16 = df_16['relapse']
relapse_rate_20 = df_20['relapse']
relapse_rate_24 = df_24['relapse']
relapse_rates = [relapse_rate_16, relapse_rate_20, relapse_rate_24]

# %%
def plot_histogram_relapse_rate():
    plt.figure(figsize=(12, 8))
    
    plt.hist(df_16[target], bins=20, alpha=0.5, label='16 Months', color='b')
    plt.hist(df_20[target], bins=20, alpha=0.5, label='20 Months', color='g')
    plt.hist(df_24[target], bins=20, alpha=0.5, label='24 Months', color='r')
    
    plt.xlabel('Relapse Rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of Relapse Rates by Timeframe')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

plot_histogram_relapse_rate()

# %%
def plot_ite_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    data_16 = [ite_syncf_16]
    data_20 = [ite_syncf_20]
    data_24 = [ite_syncf_24]
    
    models = ['Counterfactual Synthetic RF']
    
    axes[0].boxplot(data_16)
    axes[0].set_title('ITE Distribution (16 months)')
    axes[0].set_xticklabels(models, rotation=45)
    
    axes[1].boxplot(data_20)
    axes[1].set_title('ITE Distribution (20 months)')
    axes[1].set_xticklabels(models, rotation=45)
    
    axes[2].boxplot(data_24)
    axes[2].set_title('ITE Distribution (24 months)')
    axes[2].set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
plot_ite_distributions()

# %%
def plot_ite_distribution_bar():
    data = [ite_syncf_16, ite_syncf_20, ite_syncf_24]
    labels = ['16 months', '20 months', '24 months']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[0], bins=30, color='blue', label='16 months', kde=True)
    sns.histplot(data[1], bins=30, color='green', label='20 months', kde=True)
    sns.histplot(data[2], bins=30, color='red', label='24 months', kde=True)
    
    plt.legend()
    plt.title('ITE Distribution for Counterfactual Synthetic RF')
    plt.xlabel('ITE Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('ite_distribution_synct_cf.png')  # Save the figure
    plt.show()

plot_ite_distribution_bar()

# %%
def plot_ite_correlation_heatmap():
    # Compute correlation matrix
    data = {
        '16 months': ite_syncf_16[:21],
        '20 months': ite_syncf_20[:21],
        '24 months': ite_syncf_24
    }
    df = pd.DataFrame(data)
    corr_matrix = df.corr()

    # Plot heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', cbar_kws={'shrink': .8})
    plt.title('ITE Correlation Heatmap (Counterfactual Synthetic RF)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_ite_correlation_heatmap()

# %%
def plot_ite_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    data_16 = [ite_vt_16, ite_cf_16, ite_syncf_16]
    data_20 = [ite_vt_20, ite_cf_20, ite_syncf_20]
    data_24 = [ite_vt_24, ite_cf_24, ite_syncf_24]
    
    models = ['Virtual Twins', 'Counterfactual RF', 'Counterfactual Synthetic RF']
    
    axes[0].boxplot(data_16)
    axes[0].set_title('ITE Distribution (16 months)')
    axes[0].set_xticklabels(models, rotation=45)
    
    axes[1].boxplot(data_20)
    axes[1].set_title('ITE Distribution (20 months)')
    axes[1].set_xticklabels(models, rotation=45)
    
    axes[2].boxplot(data_24)
    axes[2].set_title('ITE Distribution (24 months)')
    axes[2].set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
plot_ite_distributions()

# %%
def plot_mean_ite_trend():
    models = ['Virtual Twins', 'Counterfactual RF', 'Counterfactual Synthetic RF']
    timeframes = [16, 20, 24]
    
    mean_ite_vt = [np.mean(ite_vt_16), np.mean(ite_vt_20), np.mean(ite_vt_24)]
    mean_ite_cf = [np.mean(ite_cf_16), np.mean(ite_cf_20), np.mean(ite_cf_24)]
    mean_ite_syncf = [np.mean(ite_syncf_16), np.mean(ite_syncf_20), np.mean(ite_syncf_24)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(timeframes, mean_ite_vt, marker='o', label='Virtual Twins')
    plt.plot(timeframes, mean_ite_cf, marker='s', label='Counterfactual RF')
    plt.plot(timeframes, mean_ite_syncf, marker='^', label='Counterfactual Synthetic RF')
    
    plt.xlabel('Timeframe (months)')
    plt.ylabel('Mean ITE')
    plt.title('Mean ITE Trend Across Timeframes')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

plot_mean_ite_trend()

# %%
def plot_ite_correlation_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (ite_vt, ite_cf, ite_syncf, timeframe) in enumerate([(ite_vt_16, ite_cf_16, ite_syncf_16, 16),
                                                                 (ite_vt_20, ite_cf_20, ite_syncf_20, 20),
                                                                 (ite_vt_24, ite_cf_24, ite_syncf_24, 24)]):
        corr_matrix = np.corrcoef([ite_vt, ite_cf, ite_syncf])
        im = axes[idx].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[idx].set_title(f'ITE Correlation ({timeframe} months)')
        axes[idx].set_xticks([0, 1, 2])
        axes[idx].set_yticks([0, 1, 2])
        axes[idx].set_xticklabels(['VT', 'CF', 'SynCF'], rotation=45)
        axes[idx].set_yticklabels(['VT', 'CF', 'SynCF'])
        
        for i in range(3):
            for j in range(3):
                axes[idx].text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color='black')
        
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

plot_ite_correlation_heatmap()

# %% [markdown]
# # HyperParameter Tunning

# %%
from sklearn.model_selection import GridSearchCV

class VirtualTwins:
    def __init__(self, include_interactions=True):
        self.include_interactions = include_interactions
        self.rf = RandomForestRegressor(random_state=42, oob_score=True)
        self.param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_features': [2, 5, 7],
            'min_samples_leaf': [1, 3, 5]
        }
        self.grid_search = GridSearchCV(estimator=self.rf, param_grid=self.param_grid, cv=5, scoring='neg_mean_squared_error')
    
    def fit(self, X, T, y):
        if self.include_interactions:
            X_with_interactions = np.column_stack((X, T, X * T[:, np.newaxis]))
        else:
            X_with_interactions = np.column_stack((X, T))
        self.grid_search.fit(X_with_interactions, y)
        self.best_rf = self.grid_search.best_estimator_
        print(f"Best Virtual Twins Model: {self.grid_search.best_params_}")
        print(f"Best Virtual Twins OOB Score: {self.best_rf.oob_score_}")

    def predict_ite(self, X):
        if self.include_interactions:
            X_t1 = np.column_stack((X, np.ones(X.shape[0]), X))
            X_t0 = np.column_stack((X, np.zeros(X.shape[0]), np.zeros_like(X)))
        else:
            X_t1 = np.column_stack((X, np.ones(X.shape[0])))
            X_t0 = np.column_stack((X, np.zeros(X.shape[0])))

        y_t1 = self.best_rf.predict(X_t1)
        y_t0 = self.best_rf.predict(X_t0)

        return y_t1 - y_t0

# %%
class CounterfactualRF:
    def __init__(self):
        self.param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_features': [2, 5, 7],
            'min_samples_leaf': [1, 3, 5]
        }
        self.grid_search_t1 = GridSearchCV(estimator=RandomForestRegressor(random_state=42, oob_score=True), param_grid=self.param_grid, cv=5, scoring='neg_mean_squared_error')
        self.grid_search_t0 = GridSearchCV(estimator=RandomForestRegressor(random_state=42, oob_score=True), param_grid=self.param_grid, cv=5, scoring='neg_mean_squared_error')

    def fit(self, X, T, y):
        self.grid_search_t1.fit(X[T == 1], y[T == 1])
        self.best_rf_t1 = self.grid_search_t1.best_estimator_
        print(f"Best Counterfactual RF (T=1) Model: {self.grid_search_t1.best_params_}")
        print(f"Best Counterfactual RF (T=1) OOB Score: {self.best_rf_t1.oob_score_}")

        self.grid_search_t0.fit(X[T == 0], y[T == 0])
        self.best_rf_t0 = self.grid_search_t0.best_estimator_
        print(f"Best Counterfactual RF (T=0) Model: {self.grid_search_t0.best_params_}")
        print(f"Best Counterfactual RF (T=0) OOB Score: {self.best_rf_t0.oob_score_}")

    def predict_ite(self, X):
        y_t1 = self.best_rf_t1.predict(X)
        y_t0 = self.best_rf_t0.predict(X)
        return y_t1 - y_t0

# %%
def train_tuned_models(t):
    (X_train, X_test, T_train, T_test, y_train, y_test) = split_data(t)

    vt = VirtualTwins(include_interactions=True)
    vt.fit(X_train, T_train, y_train)
    ite_vt = vt.predict_ite(X_test)

    cf_rf = CounterfactualRF()
    cf_rf.fit(X_train, T_train, y_train)
    ite_cf = cf_rf.predict_ite(X_test)

    syncf = CounterfactualSyntheticRF()
    syncf.fit(X_train, T_train, y_train)
    ite_syncf = syncf.predict_ite(X_test)

    rmse_vt = calculate_ite_rmse(ite_vt, y_test, T_test)
    rmse_cf = calculate_ite_rmse(ite_cf, y_test, T_test)
    rmse_syncf = calculate_ite_rmse(ite_syncf, y_test, T_test)

    return (ite_vt, ite_cf, ite_syncf, rmse_vt, rmse_cf, rmse_syncf)

# %%
tuned_results_16 = train_tuned_models(16)
tuned_results_20 = train_tuned_models(20)
tuned_results_24 = train_tuned_models(24)

# %%
def compare_tuned_ite_statistics(results_16, results_20, results_24):
    ite_vt_16, ite_cf_16, ite_syncf_16 = results_16[:3]
    ite_vt_20, ite_cf_20, ite_syncf_20 = results_20[:3]
    ite_vt_24, ite_cf_24, ite_syncf_24 = results_24[:3]

    stats = {
        '16 months': {
            'Tuned Virtual Twins': ite_vt_16,
            'Tuned Counterfactual RF': ite_cf_16,
            'Synthetic CF': ite_syncf_16
        },
        '20 months': {
            'Tuned Virtual Twins': ite_vt_20,
            'Tuned Counterfactual RF': ite_cf_20,
            'Synthetic CF': ite_syncf_20
        },
        '24 months': {
            'Tuned Virtual Twins': ite_vt_24,
            'Tuned Counterfactual RF': ite_cf_24,
            'Synthetic CF': ite_syncf_24
        }
    }

    for timeframe in stats:
        for model in stats[timeframe]:
            ite = stats[timeframe][model]
            stats[timeframe][model] = {
                'Mean': np.mean(ite),
                'Median': np.median(ite),
                'Std Dev': np.std(ite),
                'Min': np.min(ite),
                'Max': np.max(ite),
                '25th Percentile': np.percentile(ite, 25),
                '75th Percentile': np.percentile(ite, 75)
            }

    df_stats = pd.DataFrame.from_dict({(i,j): stats[i][j] 
                                       for i in stats.keys() 
                                       for j in stats[i].keys()},
                                      orient='index')

    print("Tuned Models ITE Statistics Comparison:")
    print(df_stats.to_string())

    print("\nComparative Insights for Tuned Models:")
    for metric in ['Mean', 'Median', 'Std Dev']:
        print(f"\n{metric} ITE Comparison:")
        for timeframe in stats:
            print(f"  {timeframe}:")
            values = [stats[timeframe][model][metric] for model in stats[timeframe]]
            best_model = list(stats[timeframe].keys())[np.argmax(values) if metric != 'Std Dev' else np.argmin(values)]
            print(f"    Best model: {best_model} ({max(values) if metric != 'Std Dev' else min(values):.4f})")

    consistency_scores = {model: np.std([stats[timeframe][model]['Mean'] for timeframe in stats]) 
                          for model in stats['16 months']}
    most_consistent_model = min(consistency_scores, key=consistency_scores.get)
    print(f"\nMost consistent model across timeframes: {most_consistent_model}")
    print(f"Consistency score (lower is better): {consistency_scores[most_consistent_model]:.4f}")

compare_tuned_ite_statistics(tuned_results_16, tuned_results_20, tuned_results_24)

# %%
def plot_tuned_rmse_comparison(results_16, results_20, results_24):
    models = ['Virtual Twins', 'Counterfactual RF', 'Synthetic CF']
    rmse_16 = results_16[3:]
    rmse_20 = results_20[3:]
    rmse_24 = results_24[3:]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, rmse_16, width, label='16 months', alpha=0.7)
    rects2 = ax.bar(x, rmse_20, width, label='20 months', alpha=0.7)
    rects3 = ax.bar(x + width, rmse_24, width, label='24 months', alpha=0.7)

    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison for Tuned Models Across Timeframes')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_tuned_rmse_comparison(tuned_results_16, tuned_results_20, tuned_results_24)

# %%
def plot_tuned_ite_distributions(results_16, results_20, results_24):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    data_16 = results_16[:3]
    data_20 = results_20[:3]
    data_24 = results_24[:3]
    
    models = ['Virtual Twins', 'Counterfactual RF', 'Synthetic CF']
    
    axes[0].boxplot(data_16)
    axes[0].set_title('ITE Distribution for Tuned Models (16 months)')
    axes[0].set_xticklabels(models, rotation=45)
    
    axes[1].boxplot(data_20)
    axes[1].set_title('ITE Distribution for Tuned Models (20 months)')
    axes[1].set_xticklabels(models, rotation=45)
    
    axes[2].boxplot(data_24)
    axes[2].set_title('ITE Distribution for Tuned Models (24 months)')
    axes[2].set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    plt.show()

plot_tuned_ite_distributions(tuned_results_16, tuned_results_16, tuned_results_16)

# %% [markdown]
# # Comparision before and after tunned results

# %%
def plot_rmse_comparison():
    timeframes = ['16 months', '20 months', '24 months']
    vt_before = [rmse_vt_16, rmse_vt_20, rmse_vt_24]
    vt_after = [tuned_results_16[3], tuned_results_20[3], tuned_results_24[3]]
    cf_before = [rmse_cf_16, rmse_cf_20, rmse_cf_24]
    cf_after = [tuned_results_16[4], tuned_results_20[4], tuned_results_24[4]]

    x = np.arange(len(timeframes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, vt_before, width, label='VT Before', alpha=0.7)
    ax.bar(x - 0.5*width, vt_after, width, label='VT After', alpha=0.7)
    ax.bar(x + 0.5*width, cf_before, width, label='CF Before', alpha=0.7)
    ax.bar(x + 1.5*width, cf_after, width, label='CF After', alpha=0.7)

    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison Before and After Tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(timeframes)
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_rmse_comparison()

# %%
def plot_ite_distributions():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    timeframes = ['16 months', '20 months', '24 months']
    
    for i, (before_vt, before_cf, after_vt, after_cf) in enumerate([
        (ite_vt_16, ite_cf_16, tuned_results_16[0], tuned_results_16[1]),
        (ite_vt_20, ite_cf_20, tuned_results_20[0], tuned_results_20[1]),
        (ite_vt_24, ite_cf_24, tuned_results_24[0], tuned_results_24[1])
    ]):
        sns.boxplot(data=[before_vt, after_vt], ax=axes[0, i])
        axes[0, i].set_title(f'Virtual Twins ITE Distribution ({timeframes[i]})')
        axes[0, i].set_xticklabels(['Before', 'After'])
        
        sns.boxplot(data=[before_cf, after_cf], ax=axes[1, i])
        axes[1, i].set_title(f'Counterfactual RF ITE Distribution ({timeframes[i]})')
        axes[1, i].set_xticklabels(['Before', 'After'])
    
    plt.tight_layout()
    plt.show()

plot_ite_distributions()

# %%
def print_statistics():
    for timeframe, before_vt, before_cf, after_vt, after_cf in [
        ('16 months', ite_vt_16, ite_cf_16, tuned_results_16[0], tuned_results_16[1]),
        ('20 months', ite_vt_20, ite_cf_20, tuned_results_20[0], tuned_results_20[1]),
        ('24 months', ite_vt_24, ite_cf_24, tuned_results_24[0], tuned_results_24[1])
    ]:
        print(f"\n{timeframe}:")
        print("Virtual Twins:")
        print(f"  Before - Mean: {np.mean(before_vt):.4f}, Std Dev: {np.std(before_vt):.4f}")
        print(f"  After  - Mean: {np.mean(after_vt):.4f}, Std Dev: {np.std(after_vt):.4f}")
        print("Counterfactual RF:")
        print(f"  Before - Mean: {np.mean(before_cf):.4f}, Std Dev: {np.std(before_cf):.4f}")
        print(f"  After  - Mean: {np.mean(after_cf):.4f}, Std Dev: {np.std(after_cf):.4f}")

print_statistics()

# %%



