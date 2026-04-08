import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy as sp



url_1 = "https://docs.google.com/spreadsheets/d/1D7N1J68eW5KYGS8U1wKdFPIWddxm600svG9pXHbGMuM/edit?usp=sharing"
csv_url_1 = url_1.replace("/edit?usp=sharing", "/export?format=csv")
df_1 = pd.read_csv(csv_url_1)
df_1['Week of'] = pd.to_datetime(df_1['Week of'])

url_2 = "https://docs.google.com/spreadsheets/d/1S5E-DAx-lhk_oCIY3BdEeQd38PT6PtxiFc0BFXzOCzE/edit?usp=sharing"
csv_url_2 = url_2.replace("/edit?usp=sharing", "/export?format=csv")
intro = pd.read_csv(csv_url_2)
intro['Count'] = intro['Count'].str.replace(',', '').astype(int)
intro['Date'] = pd.to_datetime(intro['Date'])
intro.set_index('Date', inplace=True)
df_2 = intro['Count'].resample('W').mean().reset_index(name='Count')
df_2 = df_2.sort_values('Date', ascending=True).reset_index(drop=True)
df_2['Count_smooth'] = df_2['Count'].rolling(window=4, center=True, min_periods=1).mean()

print(df_1.head())
print(df_2.head())

#linregression

df_merged = pd.merge_asof(df_1.sort_values('Week of'), df_2.sort_values('Date'), left_on='Week of', right_on='Date')
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(df_merged['Price Per Gallon'], df_merged['Count'])
r_squared = r_value**2
print(f"R-squared: {r_squared:.4f}, p-value: {p_value:.4f}")

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(df_1['Week of'], df_1['Price Per Gallon'], label='Gas Prices', color='dodgerblue')  
ax1.set_ylabel('Gas Price ($/gal)', color='dodgerblue')
ax1.tick_params(axis='y', labelcolor='dodgerblue')

ax2 = ax1.twinx()
ax2.plot(df_2['Date'], df_2['Count'] / 1_000_000, label='Avg Ridership (millions)', color='#d04050', alpha=0.3)
ax2.plot(df_2['Date'], df_2['Count_smooth'] / 1_000_000, label='Ridership (4-week)', color='#d04050', linewidth=2)
ax2.set_ylabel('Subway Ridership (millions)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('NYC Gas Prices and Average Subway Ridership', color='white')
ax1.set_xlabel('Date')
ax1.grid(alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6,
           handlelength=1, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
ax1.text(0.375, 0.95, f"R² = {r_squared:.4f}\np = {p_value:.4f}", transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')

ax1.tick_params(colors='white')
ax2.tick_params(colors='white')
ax1.xaxis.label.set_color('white')
ax1.yaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')

plt.tight_layout()

plt.savefig('gas_prices_and_subway_ridership.jpg', dpi=300, bbox_inches='tight')
plt.show(block=False)


#actual OLS/log-log
df_merged = pd.merge_asof(df_1.sort_values('Week of'), df_2.sort_values('Date'), 
                          left_on='Week of', right_on='Date').dropna()
df_merged['ln_ridership'] = np.log(df_merged['Count'])
df_merged['ln_gas_price'] = np.log(df_merged['Price Per Gallon'])

df_merged['trend'] = np.arange(len(df_merged))

X = df_merged[['ln_gas_price', 'trend']]
X = sm.add_constant(X) 
y = df_merged['ln_ridership']

model = sm.OLS(y, X).fit(cov_type='HC3')

print(model.summary())

plt.figure(figsize=(10, 5))
plt.scatter(model.predict(), model.resid, alpha=0.5, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (Testing for Homoskedasticity)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Errors)')
plt.show()
