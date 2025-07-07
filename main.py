import csv
import random
import datetime

def simulate_row():
    sm = round(random.uniform(10, 60), 1)       # soil moisture %
    temp = round(random.uniform(15, 40), 1)     # air temp °C
    hum = round(random.uniform(20, 90), 1)      # air humidity %
    wind = round(random.uniform(0, 20), 1)      # wind speed km/h
    time = random.randint(0, 23)                # hour of day
    rain = random.choice([0, 1])                # rain next hour (simulated)
    
    # Rule-based watering logic
    # If soil dry (<30%) and hot (>30°C) and no rain → water
    if sm < 30 and temp > 30 and rain == 0:
        water = round(random.uniform(150, 300), 1)
        label = 1
    # If moist (>40%) or rain incoming → no water
    elif sm > 40 or rain == 1:
        water = 0.0
        label = 0
    else:
        water = round(random.uniform(50, 150), 1)
        label = 1
    
    return [sm, temp, hum, wind, time, rain, water, label]

# Generate 1000 rows
with open('irrigation_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['soil_moisture', 'temperature', 'air_humidity', 'wind_speed',
                     'time_of_day', 'rain_next_hour', 'water_given_ml', 'water_needed'])
    for _ in range(1000):
        writer.writerow(simulate_row())

print("CSV generated: irrigation_data.csv")
