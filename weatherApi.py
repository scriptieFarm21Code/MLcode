import json
import requests
from datetime import datetime

# Convert the timestamp to unix for function
def convertTimeStap(timestamp):

    utc_time = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")

    return (utc_time - datetime(1970, 1, 1)).total_seconds()
    pass

# Connects to weather api and  give  back needed data


def getWeatherData(measuredAt, lat, lon):

    timeInUnix = int(convertTimeStap(measuredAt))

    url = 'https://api.darksky.net/forecast/{}/{},{},{}?units=uk2'.format(weatherAPiKey, lat, lon, timeInUnix)
    postRequestText = requests.get(url).text
    requestsJson = json.loads(postRequestText)
    print(requestsJson)
    print(requestsJson['currently'])

    return {
        'temp': requestsJson['currently']['temperature'],
        'airPressure': requestsJson['currently']['pressure'],
        'windSpeed': (requestsJson['currently']['windSpeed'] * 1.609344),
        'rain': requestsJson['currently']['precipIntensity'],
        'windBearing': requestsJson['currently']['windBearing'],
        'cloudCover': requestsJson['currently']['cloudCover'],
        'uvIndex': requestsJson['currently']['uvIndex'],
        'visibility': requestsJson['currently']['visibility'],
        'dewPoint': requestsJson['currently']['dewPoint'],
        'humidity': requestsJson['currently']['humidity'],
        'apparentTemperature': requestsJson['currently']['apparentTemperature']
    }
    pass
