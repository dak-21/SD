
import pandas as pd
from geopy.distance import geodesic

class HospitalSurgeReadinessAgent:
    def __init__(self, hospital_excel_path):
        self.df = pd.read_excel(hospital_excel_path).copy()
        self.df.columns = self.df.columns.str.strip()

    def event_to_requirements(self, crowd_status):
        if crowd_status == "Calm":
            return None
        elif crowd_status == "Dispersing":
            return {"Beds Available": 2, "ICU Beds Available": 1, "Oxygen Cylinders Available": 1}
        elif crowd_status == "Aggressive":
            return {"Beds Available": 4, "ICU Beds Available": 2, "Oxygen Cylinders Available": 2}
        elif crowd_status == "Stampede":
            return {"Beds Available": 8, "ICU Beds Available": 4, "Oxygen Cylinders Available": 4}
        else:
            return {"Beds Available": 1}

    def recommend_hospitals(self, event_status, lat, lon, top_n=3):
        req = self.event_to_requirements(event_status)
        if not req:
            return []

        # apply resource requirements
        candidates = self.df.copy()
        for col, val in req.items():
            candidates = candidates[candidates[col] >= val]

        if candidates.empty:
            return []

        # add distance column
        candidates['DistanceMeters'] = candidates.apply(
            lambda row: geodesic((lat, lon), (row['Latitude'], row['Longitude'])).meters, axis=1
        )
        candidates = candidates.sort_values('DistanceMeters').head(top_n)

        hospitals = []
        for _, row in candidates.iterrows():
            hospitals.append({
                "Hospital Name": row['Hospital Name'],
                "Address": row['Address'],
                "City": row['City'],
                "State": row['State'],
                "Distance (meters)": round(row['DistanceMeters'], 1),
                "Total Beds": int(row['Total Beds']),
                "Beds Available": int(row['Beds Available']),
                "ICU Beds": int(row['ICU Beds']),
                "ICU Beds Available": int(row['ICU Beds Available']),
                "Oxygen Cylinders Available": int(row['Oxygen Cylinders Available']),
                "Ambulance Count": int(row['Ambulance Count']),
                "Emergency Contact": row['Emergency Contact']
            })
        return hospitals

    def notify_hospitals(self, event_status, lat, lon):
        hospitals = self.recommend_hospitals(event_status, lat, lon, top_n=3)
        if not hospitals:
            print("No hospitals with required resources found nearby.")
            return

        for hosp in hospitals:
            notification = (
                f"ALERT to {hosp['Hospital Name']} ({hosp['Distance (meters)']} m away):\n"
                f"Situation: {event_status}\n"
                f"Location: {lat}, {lon}\n"
                f"Beds Available: {hosp['Beds Available']} / Total: {hosp['Total Beds']}\n"
                f"ICU Beds Available: {hosp['ICU Beds Available']} / Total: {hosp['ICU Beds']}\n"
                f"Oxygen Cylinders Available: {hosp['Oxygen Cylinders Available']}\n"
                f"Ambulances: {hosp['Ambulance Count']}\n"
                f"Address: {hosp['Address']}, {hosp['City']}, {hosp['State']}\n"
                f"CONTACT: {hosp['Emergency Contact']}\n"
                "Please prepare paramedics and necessary resources for potential incoming cases."
            )
            print(notification)
            print("="*80)

# Example usage:
if __name__ == '__main__':
    agent = HospitalSurgeReadinessAgent('hospital_data.xlsx')
    # Example: crowd event at Marine Drive, Mumbai
    crowd_status = "Stampede"  # Change to Calm/Dispersing/Aggressive/Stampede
    static_lat = 18.9500
    static_lon = 72.8258

    agent.notify_hospitals(crowd_status, static_lat, static_lon)
