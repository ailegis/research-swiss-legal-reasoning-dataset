import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


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


def generate_visual_report(df: pd.DataFrame , save_path='results'):
    # Ensure 'Date' column is datetime, then extract 'Year'
    if 'Date' in df.columns:
        df['Year'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.year
        df = df.dropna(subset=['Year']).astype({'Year': int})

    # Initialize the figure and grid
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Distribution of Questions Across Different Categories')
    gs = GridSpec(2, 3, figure=fig, width_ratios=[3, 3, 1])

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

    # Add a combined legend with smaller size and split into two lines
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis('off')
    ax_legend.legend(
        labels=['Questions by Course', 'Questions by Year', 'Unique Questions by Course',
                'Questions by Language', 'Questions with and without Facts'],
        title="Legend", loc="center", bbox_to_anchor=(0.5, 0.5),
        fontsize='small', title_fontsize='medium', ncol=1
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(save_path+"/multichart.pdf", format='pdf')
    fig.savefig(save_path+"/multichart.png", format='png')
    plt.show()