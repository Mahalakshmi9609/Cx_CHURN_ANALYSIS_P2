"""Step 1 - Exploratory Data Analysis (EDA).

Loads the raw CSV, prints basic diagnostics, and saves distribution plots to
the configured output directory.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATA_RAW, OUTPUT_DIR, TARGET


def main():
	"""Load the data, print diagnostics, and save distribution plots."""
	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(DATA_RAW)

	print("=" * 60)
	print("SHAPE:", df.shape)
	print("=" * 60)

	print("\nCOLUMN TYPES:")
	print(df.dtypes)

	print("\nFIRST 5 ROWS:")
	print(df.head())

	print("\nDESCRIPTIVE STATISTICS:")
	print(df.describe())

	print("\nMISSING VALUES:")
	print(df.isnull().sum())

	print(f"\nTARGET DISTRIBUTION ({TARGET}):")
	print(df[TARGET].value_counts())
	print(df[TARGET].value_counts(normalize=True).round(4))

	numeric_columns = df.select_dtypes(include="number").columns
	if len(numeric_columns) > 0:
		df[numeric_columns].hist(figsize=(14, 10), bins=20)
		plt.suptitle("Numeric Feature Distributions")
		plt.tight_layout()
		plt.savefig(output_dir / "numeric_distributions.png", dpi=150)
		plt.close()

	categorical_columns = df.select_dtypes(exclude="number").columns
	for column in categorical_columns:
		plt.figure(figsize=(8, 5))
		sns.countplot(data=df, x=column, order=df[column].value_counts().index)
		plt.title(f"Distribution of {column}")
		plt.xticks(rotation=30, ha="right")
		plt.tight_layout()
		plt.savefig(output_dir / f"{column.lower().replace(' ', '_')}_distribution.png", dpi=150)
		plt.close()


if __name__ == "__main__":
	main()
