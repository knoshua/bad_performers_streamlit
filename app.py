import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def main():
    st.title('Bad Performer Exits Analysis')
    df = get_validator_data()
    
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            metric_choice = st.radio(
                    "Metric", ["Head Timeliness", "Target Timeliness", "Source Timeliness", "Weighted Timeliness"])

        with c2:
            effectiveness_target = st.slider("Effictiveness Target", 0.0, 1.0, 0.9, 0.01)

        with c3:
            sensitivity = st.slider("Sensitivity", 0.0, 1.0, 0.95, 0.01)

    metric_threshold = df.loc[
        df["Effectiveness"] < effectiveness_target, metric_choice].quantile(sensitivity)

    fig, confusion_df = get_fig_and_confusion(df.copy(), metric_choice, metric_threshold, effectiveness_target)

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

    st.write(f"**{metric_choice} threshold:** {metric_threshold:.1%}")

    st.header('Confusion Matrix')
    st.dataframe(confusion_df)

    with st.expander("Raw Data"):
        st.dataframe(df)

def get_validator_data():
    df = pd.read_pickle('rocketpool_data.pkl')
    df = df.drop(columns=[
        'total_timely_head',
        'total_timely_target',
        'total_timely_source',
        'total_earned',
        'total_missed',
        'total_penalties',
        ])
    df = df.rename(columns={
        'val_id': 'Validator ID',
        'total_epochs': 'Epochs',
        'head_frequency': 'Head Timeliness',
        'target_frequency': 'Target Timeliness',
        'source_frequency': 'Source Timeliness',
        'effectiveness': 'Effectiveness',
    })
    df["Weighted Timeliness"] = weighted_timeliness(df["Head Timeliness"], df["Target Timeliness"], df["Source Timeliness"])
    return df

def weighted_timeliness(h, t, s):
    return h + (t-h) * (40/54) - (s-t) * (12/54) - (1-s) * (40/54)

def get_fig_and_confusion(df, metric_choice, metric_threshold, effectiveness_target):
    df["flagged"] =df[metric_choice] <= metric_threshold
    df["status"] =df["flagged"].map({True: "Force Exit", False: "No Exit"})


    fig = px.scatter(
        df,
        x=df.index,
        y="Effectiveness",
        color="status",
        hover_data=["Validator ID", "Effectiveness", metric_choice],
    )
        
    fig.update_yaxes(range=[0.7, 1.0], tickformat=".0%")
    fig.update_layout(xaxis_title="Validator")

    fig.add_hline(
        y=effectiveness_target,
        line_dash="dash",
    )

    df["Effectiveness"] = np.where(
        df["Effectiveness"] < effectiveness_target,
        "Bad Performer",
        "Good Performer"
    )
    confusion_df = confusion_matrix(df, 'Effectiveness', 'status').style.apply(highlight_positive_diagonal, axis=None)
    return fig, confusion_df

def confusion_matrix(df: pd.DataFrame,actual: str, predicted: str):
    return (
            df
            .groupby([actual, predicted])
            .size()
            .unstack(fill_value=0)
            )

def highlight_positive_diagonal(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    n_rows, n_cols = df.shape
    n = min(n_rows, n_cols)

    for i in range(n):
        styles.iat[i, n_cols - 1 - i] = 'background-color: #ffeb99; color: black; font-weight: bold;'

    return styles

if __name__ == "__main__":
    main()
