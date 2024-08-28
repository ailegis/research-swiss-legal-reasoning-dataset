import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

from feature_extraction import parseToListFunc


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

    # Generate the report table with special handling for unhashable types
    def safe_nunique(col):
        try:
            return col.nunique()
        except TypeError:
            return "Unhashable data type"

    def unique_values_example(col):
        try:
            if col.nunique() < 10:
                return ", ".join(map(str, col.unique()))
            else:
                return "Too many"
        except TypeError:
            return "Unhashable data type"
    report_df = pd.DataFrame({
        "Column": df.columns,
        "Unique Values Count": df.apply(safe_nunique),
        "Examples of Unique Values": df.apply(unique_values_example)
    })
    print("Summary of Unique Values per Column:\n", report_df.to_string(index=False), "\n")

    print("Course List")
    print(df["Course"].unique(), "\n")

    # Examples: Display the first few rows of the dataframe
    print("First few rows of the data:")
    print(df.head(), "\n")



def preprocess_data(df):
    if 'Date' in df.columns:
        df['Year'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.year
        df = df.dropna(subset=['Year']).astype({'Year': int})
    df['Facts_Category'] = df['Facts'].apply(lambda x: 'With Facts' if len(str(x)) > 2 else 'No Facts')
    df['Use_Cit_Question'] = df['QuestionCitations'].apply(lambda x: 'Cit.' if (len(parseToListFunc(x)) > 0) else 'No Cit.')
    df['Use_Cit_Answer'] = df['AnswerCitation'].apply(lambda x: 'Cit.' if (len(parseToListFunc(x)) > 0) else 'No Cit.')
    df['Use_Cit_Facts'] = df['FactCitation'].apply(lambda x: 'Cit.' if (len(parseToListFunc(x)) > 0) else 'No Cit.')
    df['Number_Cit_Answer'] = df['AnswerCitation'].apply(lambda x: len(parseToListFunc(x)))
    return df


def plot_pie(ax, data, title, colors, show_labels=True):
    # Save the original labels for legend creation
    labels = data.index

    # Determine whether to display labels on the pie chart
    pie_labels = labels if show_labels else None

    # Plot the pie chart with or without labels based on the parameter
    wedges, texts, autotexts = ax.pie(data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)

    # Remove labels and adjust text rotation for small slices
    for wedge, text, autotext in zip(wedges, texts, autotexts):
        percent_value = float(autotext.get_text().strip('%'))
        if percent_value < 1.5:
            autotext.set_text('')
        elif percent_value < 10:
            rotation_angle = wedge.theta2 - (wedge.theta2 - wedge.theta1) / 2
            autotext.set_rotation(rotation_angle - 360 if rotation_angle > 180 else rotation_angle)

    ax.set_title(title)
    ax.axis('equal')

    # Return the wedges and original labels for the legend
    return wedges, labels


def plot_double_pie(ax, data, inner_col, outer_col, title, course_colors):
    # White color for the inner pie with black borders
    inner_colors = ['white'] * len(data[inner_col].unique())
    inner_wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}

    # Create the inner ring based on the selected inner column
    inner_data = data.groupby(inner_col).size()
    inner_labels = inner_data.index
    
    wedges1, text1, autotexts1 = ax.pie(
        inner_data, labels=None, autopct='%1.1f%%', radius=0.7, 
        colors=inner_colors, startangle=90, wedgeprops=inner_wedgeprops
    )
    
    # Adjust autotexts1 to display within the pie
    for i, autotext in enumerate(autotexts1):
        autotext.set_text(inner_labels[i])
        autotext.set_color('black')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # Sort outer_data by inner_col and then by outer_col to ensure consistent order
    outer_data = data.groupby([inner_col, outer_col]).size().sort_index(level=[inner_col, outer_col])
    outer_labels = [f'{outer}\n({inner})' for inner, outer in outer_data.index]
    outer_colors = [course_colors[course] for _, course in outer_data.index]

    wedges2, autotexts2 = ax.pie(
        outer_data, labels=None, autopct=None, radius=1.0, 
        colors=outer_colors, startangle=90, wedgeprops=dict(width=0.3)
    )
    
    # Title and formatting
    ax.set_title(title)
    ax.axis('equal')
    
    return wedges1, inner_labels, wedges2, outer_labels


def generate_visual_report(df: pd.DataFrame, save_path='results'):
    df = preprocess_data(df)

    # Determine the top 10 courses with the most questions
    top_courses = df['Course'].value_counts().nlargest(10).index
    unique_courses = df['Course'].unique()
    top_course_colors = sns.color_palette("pastel", n_colors=len(top_courses))
    course_colors = {course: color for course, color in zip(top_courses, top_course_colors)}
    light_gray = sns.light_palette("gray", n_colors=len(unique_courses))[0]
    for course in unique_courses:
        if course not in course_colors:
            course_colors[course] = light_gray

    # Initialize the figure and grid with two rows and three columns
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle('Distribution of Questions Across Different Categories')
    gs = GridSpec(4, 3, figure=fig)

    # Define subplots and pass the ax to each plot function
    ax1 = fig.add_subplot(gs[0, 0])
    wedges_ax1, labels_ax1 = plot_pie(ax1, df['Course'].value_counts(), 'Questions by Course', 
                                      colors=[course_colors[course] for course in df['Course'].value_counts().index], 
                                      show_labels=False)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_double_pie(ax2, df, outer_col="Course", inner_col='Year', title='Questions by Year and Course', course_colors=course_colors)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_double_pie(ax3, df, outer_col="Course", inner_col='Language', title='Questions by Language', course_colors=course_colors)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    legend_labels = [course for course in top_courses]
    legend_colors = [course_colors[course] for course in top_courses]
    ax4.legend(handles=[plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10) 
                        for color in legend_colors],
               labels=legend_labels, title="Top Courses", loc='center')
    
    ax5 = fig.add_subplot(gs[1, 1])
    plot_double_pie(ax5, df, outer_col="Course", inner_col='Facts_Category', title='Questions with and without Facts', course_colors=course_colors)

    ax6 = fig.add_subplot(gs[1, 2])
    plot_double_pie(ax6, df, outer_col="Course", inner_col='QuestionType', title='Types of question format', course_colors=course_colors)

    ax7 = fig.add_subplot(gs[2, 0])
    plot_double_pie(ax7, df, outer_col="Course", inner_col='Use_Cit_Question', title='Questions with citations', course_colors=course_colors)

    ax8 = fig.add_subplot(gs[2, 1])
    plot_double_pie(ax8, df, outer_col="Course", inner_col='Use_Cit_Answer', title='Answers with citations', course_colors=course_colors)

    ax9 = fig.add_subplot(gs[2, 2])
    plot_double_pie(ax9, df, outer_col="Course", inner_col='Use_Cit_Facts', title='Facts with citations', course_colors=course_colors)

    ax10 = fig.add_subplot(gs[3, 0])
    plot_double_pie(ax10, df, outer_col="Course", inner_col='SplitCorrectness', title='Split F Q A Correctness', course_colors=course_colors)

    ax11 = fig.add_subplot(gs[3, 1])
    plot_double_pie(ax11, df, outer_col="Course", inner_col='CounterfactualAnswer', title='Counter-factual Answer', course_colors=course_colors)

    ax12 = fig.add_subplot(gs[3, 2])
    plot_double_pie(ax12, df[df["Number_Cit_Answer"] > 0], outer_col="Course", inner_col='Number_Cit_Answer', title='Number of unique citations in answer', course_colors=course_colors)



    # Adjust layout and save the figures
    fig.savefig(save_path+"/multichart.pdf", format='pdf')
    fig.savefig(save_path+"/multichart.png", format='png')


# Function to plot distribution curves
def plot_distribution_curve(ax, lengths_list, labels, title):
    # Ensure that lengths_list and labels are the same length
    assert len(lengths_list) == len(labels), "lengths_list and labels must be the same length."
    
    # Filter out lengths smaller than 1 to avoid log(0) errors
    filtered_lengths = [lengths[lengths >= 1] for lengths in lengths_list]

    # Get pastel colors from the palette
    pastel_colors = sns.color_palette("pastel", len(labels))

    # Plot the KDE plots for the normalized lengths with pastel colors
    for lengths, label, color in zip(filtered_lengths, labels, pastel_colors):
        sns.kdeplot(np.log10(lengths), ax=ax, label=label, bw_adjust=0.5, fill=True, alpha=0.5, color=color)
    
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


# Function to generate the visual report
def generate_visual_report2(df: pd.DataFrame, save_path='results'):

    columns_to_plot = ['Question', 'Facts', 'Answer']
    labels = ['Question Length', 'Fact Length', 'Answer Length']
    title = 'Distribution of Question, Fact, and Answer Lengths'


    df = preprocess_data(df)

    # Initialize the figure and grid with two rows and two columns
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Distribution of Questions Across Different Categories')
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0,0])
    
    lengths_list = [df[col].apply(lambda x: len(str(x))) for col in columns_to_plot]
    
    plot_distribution_curve(ax1, lengths_list, labels, title)

    # Adjust layout and save the figures
    fig.savefig(save_path + "/multichart2.pdf", format='pdf')
    fig.savefig(save_path + "/multichart2.png", format='png')
