# openaq_fetch.py
import requests, pandas as pd, time

def fetch_openaq_kampala(parameter="pm25", days=30):
    url = "https://api.openaq.org/v2/measurements"
    rows = []
    page = 1
    while True:
        resp = requests.get(url, params={
            "city": "Kampala",
            "parameter": parameter,
            "limit": 10000,
            "page": page,
            "sort": "desc"
        }, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results: break
        for m in results:
            rows.append({
                "timestamp_utc": m["date"]["utc"],
                "station_id": m["location"],
                "lat": m.get("coordinates", {}).get("latitude"),
                "lon": m.get("coordinates", {}).get("longitude"),
                "pm25": m["value"]
            })
        page += 1
        time.sleep(0.3)  # be polite to API
        if page > 10: break  # keep it small for exam
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df

df_hourly = fetch_openaq_kampala()

print(df_hourly.head())
df_hourly.to_csv("data/openaq_kampala_hourly.csv", index=False)
