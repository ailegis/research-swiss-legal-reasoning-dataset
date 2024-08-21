import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np


def generate_dataframe_report(df: pd.DataFrame):
    # Number of rows and columns
    rows, columns = df.shape
    print(f"The data has {rows} rows and {columns} columns.\n")

    # Column names
    print("Column names:")
    print(df.columns.tolist(), "\n")

    # Data types of each column
    print("Data types of each column:")
    print(df.dtypes, "\n")

    # Number of missing values per column
    print("Number of missing values per column:")
    print(df.isnull().sum(), "\n")

    # Generate the report table
    report_df = pd.DataFrame({
        "Column": df.columns,
        "Unique Values Count": df.nunique(),
        "Examples of Unique Values": df.apply(lambda col: ", ".join(map(str, col.unique())) if col.nunique() < 10 else "Too many")
    })
    print("Summary of Unique Values per Column:\n", report_df.to_string(index=False), "\n")

    # Examples: Display the first few rows of the dataframe
    print("First few rows of the data:")
    print(df.head(), "\n")


def generate_visual_report(df: pd.DataFrame, save_path='results'):
    # Ensure 'Date' column is datetime, then extract 'Year'
    if 'Date' in df.columns:
        df['Year'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.year
        df = df.dropna(subset=['Year']).astype({'Year': int})

    # Initialize the figure and grid with three rows
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Distribution of Questions Across Different Categories')
    gs = GridSpec(2, 3, figure=fig)

    colors = sns.color_palette("pastel")

    def plot_pie(ax, data, title, limit=10):
        if len(data) > limit:
            data = data.nlargest(limit)
            data['Other'] = data.sum() - data[data.index != 'Other'].sum()
        wedges, texts, autotexts = ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=colors)
        for wedge, text, autotext in zip(wedges, texts, autotexts):
            percent_value = float(autotext.get_text().strip('%'))
            if percent_value == 0.0:
                autotext.set_text('')
                text.set_text('')
            elif percent_value < 10:
                rotation_angle = wedge.theta2 - (wedge.theta2 - wedge.theta1) / 2
                autotext.set_rotation(rotation_angle - 360 if rotation_angle > 180 else rotation_angle)
        ax.set_title(title)
        ax.axis('equal')

    def plot_distribution_curve(ax, question_lengths, fact_lengths, title):
        # Filter out lengths smaller than 1 to avoid log(0) errors
        question_lengths = question_lengths[question_lengths >= 1]
        fact_lengths = fact_lengths[fact_lengths >= 1]

        # Check if the data is empty after normalization
        if question_lengths.empty or fact_lengths.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel('Log(Length)')
            ax.set_ylabel('Normalized Density')
            ax.grid(True, which="both", ls="--")
            return

        # Get pastel colors from the palette
        pastel_colors = sns.color_palette("pastel")

        # Plot the KDE plots for the normalized lengths with pastel colors
        sns.kdeplot(np.log10(question_lengths), ax=ax, label='Question Length', bw_adjust=0.5, fill=True, alpha=0.5, color=pastel_colors[0])
        sns.kdeplot(np.log10(fact_lengths), ax=ax, label='Fact Length', bw_adjust=0.5, fill=True, alpha=0.5, color=pastel_colors[1])
        
        # Adjust x-axis ticks for the log scale manually
        ax.set_title(title)
        ax.set_xlabel('Number of characters')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Custom ticks corresponding to lengths 1, 10, 100, 1000 on a log10 scale
        custom_ticks = [1, 2, 3, 5]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f'${pow(10, i)}$' for i in custom_ticks])
        
        ax.set_xlim(left=np.log10(1))  # Adjusted for log scale with normalization
        ax.grid(True, which="both", ls="--")

    # Plotting
    plot_pie(fig.add_subplot(gs[0, 0]), df['Course'].value_counts(), 'Questions by Course')
    if 'Year' in df.columns:
        plot_pie(fig.add_subplot(gs[0, 1]), df['Year'].value_counts(), 'Questions by Year')
    plot_pie(fig.add_subplot(gs[1, 0]), df.groupby('Course')['Question'].nunique(), 'Unique Questions by Course')
    if 'Language' in df.columns:
        plot_pie(fig.add_subplot(gs[1, 1]), df['Language'].value_counts(), 'Questions by Language')
    if 'Facts' in df.columns:
        facts_distribution = pd.Series([df['Facts'].apply(lambda x: len(str(x)) > 2).sum(), len(df) - df['Facts'].apply(lambda x: len(str(x)) > 2).sum()], 
                                       index=['With Facts', 'Without Facts'])
        plot_pie(fig.add_subplot(gs[0, 2]), facts_distribution, 'Questions with and without Facts')

    # New plot for combined question length and fact length distribution curve
    df['Question Length'] = df['Question'].apply(lambda x: len(str(x)))
    if 'Facts' in df.columns:
        df['Fact Length'] = df['Facts'].apply(lambda x: len(str(x)))
        plot_distribution_curve(fig.add_subplot(gs[1, 2:]), df['Question Length'], df['Fact Length'], 'Distribution of Question and Fact Lengths')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(save_path+"/multichart.pdf", format='pdf')
    fig.savefig(save_path+"/multichart.png", format='png')
    plt.show()