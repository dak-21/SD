import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from agent_notifier import HospitalSurgeReadinessAgent

st.markdown("""
    <style>
    body {background: #f3f6fb;}
    .stApp {background: linear-gradient(120deg, #dde2ef 0%, #c9e7ff 100%);}
    .popup {
        background: linear-gradient(90deg,#7ed6df,#70a1ff 80%);
        padding:24px;
        border-radius:12px;
        border-left:10px solid #3867d6;
        box-shadow:0 2px 16px rgba(64,128,225,0.12);
        margin-bottom:24px;
        font-size:1.2em;
        color:#204175;
    }
    .popup .big { font-size:1.7em; font-weight:700; }
    .popup .green { font-size:1.25em; color:#27ae60; }
    </style>
""", unsafe_allow_html=True)

#  MODEL/RESOURCE LOAD 
@st.cache_resource
def load_lstm_resources():
    lstm_model = load_model("lstm_crowd_behavior.h5")
    label_encoder_classes = np.load("label_encoder_classes.npy", allow_pickle=True)
    return lstm_model, label_encoder_classes

lstm_model, label_classes = load_lstm_resources()
sequence_length = 10

#Hospital AI Agent
hospital_agent = HospitalSurgeReadinessAgent('hospital_data.xlsx')

st.title("🚦 Crowd Safety & Hospital Notification Dashboard")
st.write("""
Upload crowd event data (CSV) to predict risk and send surge alerts to Mumbai hospitals.
""")

tab1, tab2 = st.tabs(["Safety Analysis & Notification", "System Control"])

with tab1:
    st.header("📈 Crowd Behavior Prediction")
    uploaded_file = st.file_uploader("Upload Crowd Event CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if all(col in df.columns for col in ["Density", "Speed", "PoseVariance"]):
            X = df[["Density", "Speed", "PoseVariance"]].values
            sequences = [X[i:i+sequence_length] for i in range(len(X) - sequence_length)]
            if sequences:
                sequences = np.array(sequences)
                preds = lstm_model.predict(sequences)
                pred_labels = label_classes[np.argmax(preds, axis=1)]
                latest_behavior = pred_labels[-1]
                st.success(f"**Predicted Crowd Behavior:** `{latest_behavior}`")

                df_show = df.assign(CrowdBehavior=["-"]*sequence_length + list(pred_labels))
                st.dataframe(df_show.tail(10), use_container_width=True)

                event_lat = float(df["Latitude"].iloc[-1]) if "Latitude" in df.columns else 18.9500
                event_lon = float(df["Longitude"].iloc[-1]) if "Longitude" in df.columns else 72.8277

                if latest_behavior != "Safe":
                    '''send this alet to superviser and then acceptance of the notif to send to the 
                    ##hospital based on superviser's action.'''
                    st.subheader("🚑 Top Mumbai Hospital Recommendations")
                    hosp_list = hospital_agent.recommend_hospitals(latest_behavior, event_lat, event_lon, top_n=3)
                    if not hosp_list:
                        st.error("No hospitals meet required resources for this emergency.")
                    else:
                        best_hosp = hosp_list[0]
                        st.markdown(
                            f"""
                            <div class="popup">
                                <span class="big">🚨 Notification Sent!</span><br>
                                Your alert has been sent to <b style="color:#1456c2;">{best_hosp["Hospital Name"]}</b>.<br>
                                <span>Hospital response teams are requested to deploy paramedics and resources to the event site.</span><br>
                                <span class="green">✅</span>
                            </div>
                            """, unsafe_allow_html=True
                        )

                        # Show all hospitals
                        for hosp in hosp_list:
                            st.markdown(
                                f"""
                                ---
                                ### 🏥 {hosp['Hospital Name']} ({hosp['Distance (meters)']} m)
                                - **Beds**: {hosp['Beds Available']} / {hosp['Total Beds']}
                                - **ICU**: {hosp['ICU Beds Available']} / {hosp['ICU Beds']}
                                - **O2:** {hosp['Oxygen Cylinders Available']} &nbsp; **Amb:** {hosp['Ambulance Count']}
                                - **Address**: {hosp['Address']}, {hosp['City']} {hosp['State']}
                                - **Contact**: {hosp['Emergency Contact']}
                                Hospital notified for surge response.
                                """
                            )
                else:
                    st.info("No emergency/dispersal risk detected. Hospitals not notified.")
                # Plots
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
                time_history = list(range(len(df)))
                ax1.plot(time_history, df["Density"], 'b-', label='Density')
                ax1.set_title('Density Over Time')
                ax1.set_xlabel('Frame'); ax1.set_ylabel('People/10k px'); ax1.legend(); ax1.grid(True)
                ax2.plot(time_history, df["Speed"], 'r-', label='Speed')
                ax2.set_title('Crowd Speed Over Time')
                ax2.set_xlabel('Frame'); ax2.set_ylabel('Pixels/Frame'); ax2.legend(); ax2.grid(True)
                ax3.plot(time_history, df["PoseVariance"], 'g-', label='Pose Var')
                ax3.set_title('Movement Variance Over Time')
                ax3.set_xlabel('Frame'); ax3.set_ylabel('Variance'); ax3.legend(); ax3.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
                st.bar_chart(pd.Series(pred_labels).value_counts())
            else:
                st.warning("Need at least {} rows for prediction.".format(sequence_length+1))
        else:
            st.error("CSV must have Density, Speed, PoseVariance columns.")

    st.header("Manual Hospital Notification")
    with st.expander("Trigger Manual Mumbai Alert"):
        manual_status = st.selectbox("Manual Crowd Status", ["Calm", "Dispersing", "Aggressive", "Stampede"])
        manual_lat = st.number_input("Event Latitude", value=18.9500, format="%.5f")
        manual_lon = st.number_input("Event Longitude", value=72.8277, format="%.5f")
        if st.button("Notify Hospitals", key="manual_notif"):
            hosp_manual = hospital_agent.recommend_hospitals(manual_status, manual_lat, manual_lon, top_n=3)
            if not hosp_manual:
                st.error("No hospitals meet required resources.")
            else:
                best_manual = hosp_manual[0]
                st.markdown(
                    f"""
                    <div class="popup">
                        <span class="big">🚨 Notification Sent!</span><br>
                        Manual alert sent to <b style="color:#1456c2;">{best_manual["Hospital Name"]}</b>.<br>
                        Teams requested for event response.<br>
                        <span class="green">✅</span>
                    </div>
                    """, unsafe_allow_html=True
                )
                for hosp in hosp_manual:
                    st.markdown(
                        f"""
                        ---
                        ### 🏥 {hosp['Hospital Name']} ({hosp['Distance (meters)']} m)
                        - **Beds**: {hosp['Beds Available']} / {hosp['Total Beds']}
                        - **ICU**: {hosp['ICU Beds Available']} / {hosp['ICU Beds']}
                        - **O2:** {hosp['Oxygen Cylinders Available']} &nbsp; **Amb:** {hosp['Ambulance Count']}
                        - **Address**: {hosp['Address']}, {hosp['City']} {hosp['State']}
                        - **Contact**: {hosp['Emergency Contact']}
                        Hospital notified for surge response.
                        """
                    )

with tab2:
    st.header("System and Threshold Settings")
    st.info("Thresholds, RL agent tuning, and admin panels.")

st.markdown("---\nCrowd Safety Analytics Dashboard - v2.0 for Supervisors")
