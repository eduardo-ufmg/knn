import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_and_process_data(filepath="results/test_results.json"):
    """
    Loads test results from a JSON file and processes them into a pandas DataFrame.

    Parameters:
        filepath (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with results for Parameterless KNN models.
            - dict: Dictionary with results for the Scikit-learn KNN wrapper.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None, None

    # Separate the scikit-learn wrapper from the other models
    sklearn_wrapper_data = data.pop("Scikit-learn KNN Wrapper", None)

    # Process the remaining models
    processed_data = []
    for model_name, results in data.items():
        # Extract the part of the name in parentheses for a shorter label
        try:
            short_name = model_name.split("(")[1].split(")")[0]
        except IndexError:
            short_name = model_name

        record = {
            "Model": short_name,
            "Accuracy": results.get("overall_accuracy"),
            "Training Time (s)": results.get("training_time"),
            "Prediction Time (s)": results.get("prediction_time"),
            "h": results.get("optimal_parameters", {}).get("h"),
            "k": results.get("optimal_parameters", {}).get("k"),
            "Reference Samples": results.get("reference_samples_used"),
        }
        processed_data.append(record)

    df = pd.DataFrame(processed_data)
    # Calculate the distance of 'h' from 1.0 for finding the best 'h'
    df["h_dist_from_1"] = (df["h"] - 1).abs()

    return df, sklearn_wrapper_data


def style_summary_table(df):
    """
    Applies styling to the summary DataFrame to highlight the best values.

    Parameters:
        df (pd.DataFrame): The DataFrame to style.

    Returns:
        pd.io.formats.style.Styler: A styled DataFrame object.
    """
    # Create a copy to avoid modifying the original DataFrame
    styled_df = df.copy()
    # We don't need this column in the final display
    if "h_dist_from_1" in styled_df.columns:
        styled_df = styled_df.drop(columns=["h_dist_from_1"])

    # Function to highlight the best values in the table
    def highlight_best(s):
        is_max = s == s.max()
        is_min = s == s.min()
        styles = []
        for v in s.index:
            style = "background-color: lightgreen"
            # For these columns, the minimum value is the best
            if s.name in [
                "Training Time (s)",
                "Prediction Time (s)",
                "Reference Samples",
            ]:
                if s[v] == s.min():
                    styles.append(style)
                else:
                    styles.append("")
            # For Accuracy, the maximum value is best
            elif s.name == "Accuracy":
                if s[v] == s.max():
                    styles.append(style)
                else:
                    styles.append("")
            else:
                styles.append("")
        return styles

    # Special handling for 'h' as its criteria is different
    def highlight_h(df_to_style):
        best_h_index = df["h_dist_from_1"].idxmin()
        styles = [""] * len(df_to_style)
        if "h" in df_to_style.columns:
            styles[best_h_index] = "background-color: lightgreen"
        return styles

    styled = styled_df.style.apply(
        highlight_best,
        subset=[
            "Accuracy",
            "Training Time (s)",
            "Prediction Time (s)",
            "Reference Samples",
        ],
    )
    styled = styled.apply(highlight_h, axis=None)
    return styled.set_caption(
        "Model Performance Summary (Best values are highlighted)"
    ).format(precision=4)


def create_visualizations(df):
    """
    Creates and displays bar plots for each performance metric.

    Parameters:
        df (pd.DataFrame): The DataFrame containing model performance data.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle("Comparative Analysis of Parameterless KNN Models", fontsize=20)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    metrics = {
        "Accuracy": {"lower_is_better": False},
        "Training Time (s)": {"lower_is_better": True},
        "Prediction Time (s)": {"lower_is_better": True},
        "h": {"special_case": "closest_to_1"},
        "Reference Samples": {"lower_is_better": True},
        "k": {},  # No highlighting for k
    }

    for i, (metric, properties) in enumerate(metrics.items()):
        ax = axes[i]

        # Determine the best model for the current metric
        best_model_idx = None
        if properties.get("lower_is_better"):
            best_model_idx = df[metric].idxmin()
        elif properties.get("lower_is_better") == False:
            best_model_idx = df[metric].idxmax()
        elif properties.get("special_case") == "closest_to_1":
            best_model_idx = df["h_dist_from_1"].idxmin()

        # Create a color palette
        palette = sns.color_palette("viridis", len(df))
        if best_model_idx is not None:
            palette[best_model_idx] = sns.color_palette("pastel")[2]

        # Create the bar plot
        sns.barplot(
            x=metric,
            y="Model",
            data=df,
            ax=ax,
            palette=palette,
            orient="h",
            hue="Model",
        )

        ax.set_title(f"Comparison of {metric}", fontsize=14)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel("Model", fontsize=12)

        # Add a star or label to highlight the best bar
        if best_model_idx is not None:
            best_bar = ax.patches[best_model_idx]
            best_bar.set_edgecolor("black")
            best_bar.set_linewidth(2)

    # Hide the last unused subplot
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def main():
    """
    Main function to run the analysis and visualization script.
    """
    # 1. Load and process the data
    df, sklearn_data = load_and_process_data()

    if df is None:
        return  # Exit if data loading failed

    # Set pandas display options for better console output
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    # 2. Display the styled summary table
    styled_table = style_summary_table(df)
    print("--- Parameterless KNN Models Performance Summary ---")
    print(df.to_string())  # Print raw data for clarity

    # Note: Displaying the styled table directly works best in environments like Jupyter
    # For a script, you might want to save it to an HTML file instead.
    # We will print the raw dataframe and then explain what the styled output would show.
    print("\n--- Styled Summary (Best Values Highlighted in Green) ---")
    print(
        "Below is what the styled table would look like. In a Jupyter Notebook, the cells would be colored."
    )

    # For demonstration in a script, we manually find and point out the bests
    print(
        f"Best Accuracy: {df.loc[df['Accuracy'].idxmax()]['Model']} ({df['Accuracy'].max():.4f})"
    )
    print(
        f"Best (Lowest) Training Time: {df.loc[df['Training Time (s)'].idxmin()]['Model']} ({df['Training Time (s)'].min():.4f}s)"
    )
    print(
        f"Best (Lowest) Prediction Time: {df.loc[df['Prediction Time (s)'].idxmin()]['Model']} ({df['Prediction Time (s)'].min():.4f}s)"
    )
    print(
        f"Best 'h' (closest to 1): {df.loc[df['h_dist_from_1'].idxmin()]['Model']} ({df.loc[df['h_dist_from_1'].idxmin()]['h']:.4f})"
    )
    print(
        f"Best (Lowest) Reference Samples: {df.loc[df['Reference Samples'].idxmin()]['Model']} ({int(df['Reference Samples'].min())})"
    )

    # 3. Report the Scikit-learn KNN Wrapper details
    if sklearn_data:
        print("\n" + "=" * 50)
        print("--- Scikit-learn KNN Wrapper (Reference) ---")
        print(f"Overall Accuracy: {sklearn_data.get('overall_accuracy'):.4f}")
        print(f"Training Time: {sklearn_data.get('training_time'):.4f}s")
        print(f"Prediction Time: {sklearn_data.get('prediction_time'):.4f}s")
        print("Optimal Parameters:")
        for key, value in sklearn_data.get("optimal_parameters", {}).items():
            print(f"  - {key}: {value}")
        print("=" * 50 + "\n")

    # 4. Generate and show the visualizations
    print("Generating visualizations...")
    create_visualizations(df)
    print("Done.")


if __name__ == "__main__":
    main()
