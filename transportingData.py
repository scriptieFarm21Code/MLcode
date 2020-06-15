# Connect to farm21 SQL database
from pymongo import MongoClient
import sys
import mysql.connector
import datetime
import json
from weatherApi import getWeatherData
# import configs
from config.connectSql import cursor
from config.connectMongo import db

# Get All the old vars
readingIDs = db.readingDataRaw.find({}, {'orginalReadingId': 1, '_id': 0})
readingIDsUsed = []

for readingObject in readingIDs:
    readingIDsUsed.append(readingObject['orginalReadingId'])
    pass

workingId = [100, 97, 98, 37, 64, 50, 48, 51]

for test in workingId:
    print(test)
    format_strings = ','.join(['%s'] * len(readingIDsUsed))

    cursor.execute(
        f"SELECT *, readings.id as readingID, soil_types.name AS soilName FROM readings JOIN sensors ON readings.sensor_id = sensors.id JOIN soil_types ON sensors.soil_type_id = soil_types.id WHERE sensor_id = {test} AND readings.measured_at > 0 AND sensors.latitude != 0 AND sensors.longitude != 0 AND soil_moisture_30 > 8 AND readings.id NOT IN (%s) AND soil_moisture_10 < 100 ORDER BY measured_at DESC LIMIT 10000" % format_strings, tuple(readingIDsUsed))

    result = cursor.fetchall()
    print(len(result))  # Print lenght to see if its enough

    for reading in result:
        print(reading)
        # Create object to save here
        weatherData = getWeatherData(reading['measured_at'], reading['latitude'], reading['longitude'])
        # Add the data from the db
        print(float(reading['soil_moisture_10']))
        print(type(weatherData))
        weatherData['soil_moisture_10'] = float(reading['soil_moisture_10'])
        weatherData['soil_moisture_20'] = float(reading['soil_moisture_20'])
        weatherData['soil_moisture_30'] = float(reading['soil_moisture_30'])
        weatherData['soil_temperature'] = float(reading['soil_temperature'])
        weatherData['lat'] = float(reading['latitude'])
        weatherData['long'] = float(reading['longitude'])
        weatherData['sensorId'] = int(reading['sensor_id'])
        weatherData['readingId'] = int(reading['readingID'])
        weatherData['sensorVersion'] = int(reading['sensor_version_id'])
        weatherData['sensorGroupId'] = int(reading['group_id'])
        weatherData['timestampReading'] = str(reading['measured_at'])
        weatherData['airTempSensor'] = float(reading['air_temperature'])
        weatherData['sensorHumidity'] = float(reading['humidity'])
        weatherData['soilType'] = str(reading['soilName'])
        weatherData['cropType'] = str(reading['crop_type'])
        weatherData['cropRace'] = str(reading['race_type'])
        weatherData['typeField'] = str(reading['cultivation_type'])
        weatherData['soilTypeId'] = int(reading['soil_type_id'])

        cursor.execute(
            f"SELECT raw.soil_moisture_10, raw.soil_moisture_20, raw.soil_moisture_30 FROM readings JOIN raw_readings AS raw ON readings.raw_reading_id = raw.id WHERE readings.id = {int(reading['readingID'])}")

        result = cursor.fetchall()

        weatherData['soil_moisture_10_raw'] = result[0]['soil_moisture_10']
        weatherData['soil_moisture_20_raw'] = result[0]['soil_moisture_20']
        weatherData['soil_moisture_30_raw'] = result[0]['soil_moisture_30']

        # print(weatherData)

        db.readingDataRaw.insert(weatherData)

        pass
