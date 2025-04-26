import matplotlib.pyplot as plt
import seaborn as sns

def pairplot_eda(df):
    sns.set_theme(style="whitegrid")
    sns.pairplot(df, hue='Exercise Recommendation Plan', palette='viridis')
    plt.savefig(fname='fig')
    plt.show()

def histograms_eda(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Body Fat Percentage'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Body Fat Percentage')
    plt.xlabel('Body Fat Percentage')
    plt.ylabel('Frequency')

    plt.figure(figsize=(10, 6))
    sns.histplot(df['BMI'], bins=30, kde=True, color='blue')
    plt.title('Distribution of BMI')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.show()

def numerical_features_distribution(df):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Distribution of Numerical Features")
    sns.histplot(df['Weight'], kde=True, ax=axes[0, 0], color='blue').set_title("Weight Distribution")
    sns.histplot(df['Height'], kde=True, ax=axes[0, 1], color='blue').set_title("Height Distribution")
    sns.histplot(df['BMI'], kde=True, ax=axes[1, 0], color='blue').set_title("BMI Distribution")
    sns.histplot(df['Body Fat Percentage'], kde=True, ax=axes[1, 1], color='blue').set_title("Body Fat Percentage Distribution")
    sns.histplot(df['Age'], kde=True, ax=axes[2, 0], color='blue').set_title("Age Distribution")
    sns.histplot(df['Exercise Recommendation Plan'], discrete=True, ax=axes[2, 1]).set_title("Exercise Recommendation Plan Distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(11, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Correlation Matrix')
    plt.show()