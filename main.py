#!/usr/bin/env python3
"""
Astronomical Visibility Calculator

This script calculates the months when a celestial object:
1. Reaches 30° above the eastern horizon at the start of astronomical twilight
2. Is at meridian at midnight

Inputs: Right Ascension (RA), Declination (Dec), Observer Latitude, Observer Longitude
Output: Month(s) of the year for each event

T CrB: --ra 239.875674 --dec 25.920224 --lat 39.2369 --lon -84.4741
CY Aqr --ra 339.449375 --dec 1.534361 --lat 39.2369 --lon -84.4741

python main.py --ra 239.875674 --dec 25.920224 --lat 39.2369 --lon -84.4741
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
from astroquery.simbad import Simbad
from astropy.coordinates import solar_system_ephemeris
import datetime
from typing import List, Tuple, Optional
import argparse
from astropy.utils.iers import conf
import pytz
from timezonefinder import TimezoneFinder
import calendar

conf.auto_download = False
conf.iers_a_file = '~/Desktop/obs-window/finals2000A.all'
import urllib.request
import ssl
context = ssl._create_unverified_context()
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=context)))
Simbad.ROW_LIMIT = 1  # Limit to one result
Simbad.TIMEOUT = 10   # Set timeout to 10 seconds
import os
pyfile_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["REQUESTS_CA_BUNDLE"] = pyfile_dir + "/ca4.cer"
import certifi

class AstronomicalVisibilityCalculator:
    """Calculate astronomical visibility events for celestial objects."""

    def __init__(self, ra: float, dec: float, target: str, latitude: float, longitude: float):
        """
        Initialize the calculator with object and observer coordinates.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            latitude: Observer latitude in degrees
            longitude: Observer longitude in degrees
        """
        self.year = 2024

        self.target = None
        if ra is not None and dec is not None:
            self.target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        elif target is not None:
            self.target = SkyCoord.from_name(target)
        self.location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
        self.latitude = latitude
        self.longitude = longitude

        # Astronomical twilight occurs when the sun is 12-18° below the horizon
        self.astronomical_twilight_angle = -12 * u.deg

        tz = TimezoneFinder()
        timezone_name = tz.timezone_at(lat=self.latitude, lng=self.longitude)
        self.timezone = pytz.timezone(timezone_name)

    def get_sun_altitude(self, time: Time) -> u.Quantity:
        """Get the sun's altitude at a given time."""
        sun = get_sun(time)
        sun_altaz = sun.transform_to(AltAz(obstime=time, location=self.location))
        return sun_altaz.alt

    def get_target_altaz(self, time: Time) -> AltAz:
        """Get the target's altitude and azimuth at a given time."""
        return self.target.transform_to(AltAz(obstime=time, location=self.location))

    def hourly_iterator(self, start_date, end_date) -> datetime:
        """
        Generates a sequence of datetime objects, incrementing by one hour
        from start_date up to (but not including) end_date.

        Args:
            start_date (datetime): The starting datetime object.
            end_date (datetime): The ending datetime object.

        Yields:
            datetime: A datetime object for each hour in the range.
        """
        current_hour = start_date
        while current_hour < end_date:
            yield current_hour
            current_hour += datetime.timedelta(hours=1)


    def find_meridian_at_midnight_months(self) -> dict:
        """
        Find months when the target object is at meridian at midnight.

        Returns:
            List of month numbers (1-12) when object is at meridian around midnight
        """
        daymonth = {"day": "", "month": ""}
        temp_alt = 30

        for month in range(1, 13):
            # Check all days
            date = datetime.date(self.year, month, 1)
            if month == 12:
                next_month = datetime.date(self.year + 1, 1, 1)
            else:
                next_month = datetime.date(self.year, month + 1, 1)
            while date < next_month:

                local_time = self.timezone.localize(datetime.datetime.strptime(str(self.year) + "-" + str(date.month) + "-" + str(date.day) + " 00:00:00", "%Y-%m-%d %H:%M:%S"))

                # Convert local time to UTC
                utc_time = local_time.astimezone(pytz.UTC)
                astropy_time = Time(utc_time)
                target_altaz = self.get_target_altaz(astropy_time)

                if target_altaz.alt > 30 * u.deg and target_altaz.alt > temp_alt * u.deg:
                    temp_alt = target_altaz.alt.deg
                    daymonth["day"] = date.day
                    daymonth["month"] = month

                date += datetime.timedelta(days=1)

        return daymonth

    def find_horizon_crossing_months(self, altitude_threshold: float) -> List[dict]:
        """
        Find months when target crosses the specified altitude at twilight.

        Args:
            altitude_threshold: Altitude in degrees (30 for the requirement)
            is_eastern: True for eastern horizon, False for western

        Returns:
            List of month numbers when the event occurs
        """
        months = []

        for month in range(1, 13):
            # Check several dates in the month
            last_day = calendar.monthrange(self.year, month)[1]

            month_num = None
            first_time_seen = None
            last_time_seen = None
            before_midnight = False
            after_midnight = False

            for day in range(1, last_day+1):
            #for day in (1, 15, last_day):
                try:
                    date = self.timezone.localize(datetime.datetime.strptime(str(self.year) + "-" + str(month) + "-" + str(day), "%Y-%m-%d"))

                    # Create a time grid for the day (every minute from 00:00 to 23:59 local time)
                    time_step = datetime.timedelta(minutes=1)

                    num_steps = int(24 * 60)  # 24 hours * 60 minutes
                    times_local = [date + i * time_step for i in range(num_steps)]
                    times_utc = [t.astimezone(pytz.UTC) for t in times_local]
                    astropy_times = Time(times_utc)

                    # Get Sun's position for all times
                    sun_coords = get_sun(astropy_times)

                    altaz_frame = AltAz(obstime=astropy_times, location=self.location)
                    sun_altaz = sun_coords.transform_to(altaz_frame)

                    # Find times when altitude is approximately 12 degrees
                    target_alt = -12.0  # degrees
                    altitudes = sun_altaz.alt.deg
                    tolerance = 1  # degrees, to account for discrete time steps

                    # Find indices where altitude is close to 12 degrees
                    close_indices = np.where(np.abs(altitudes - target_alt) < tolerance)[0]

                    if len(close_indices) > 0:
                        # Look for sign changes in (altitude - target_alt) to find exact crossings
                        diff = altitudes - target_alt
                        crossings = []
                        for i in range(len(diff) - 1):
                            if diff[i] * diff[i + 1] < 0:  # Sign change indicates crossing
                                # Linear interpolation
                                t1, t2 = times_local[i], times_local[i + 1]
                                alt1, alt2 = altitudes[i], altitudes[i + 1]
                                frac = (target_alt - alt1) / (alt2 - alt1)
                                interp_time = t1 + frac * (t2 - t1)
                                crossings.append(interp_time)

                        if len(crossings) == 2:

                            # midnight to morning
                            for hour in self.hourly_iterator(crossings[0].replace(hour=0, minute=0, second=0, microsecond=0), crossings[0]):
                                target_altaz = self.get_target_altaz(hour)
                                if target_altaz.alt.deg > altitude_threshold:
                                    month_num = month
                                    if first_time_seen is None:
                                        first_time_seen = hour
                                        after_midnight = True
                                    if last_time_seen is None or (last_time_seen is not None and last_time_seen < hour):
                                        last_time_seen = hour
                                        after_midnight = True

                            # morning astro twilight
                            target_altaz = self.get_target_altaz(crossings[0])
                            if target_altaz.alt.deg > altitude_threshold:
                                month_num = month
                                if first_time_seen is None:
                                    first_time_seen = crossings[0]
                                    after_midnight = True
                                if last_time_seen is None or (last_time_seen is not None and last_time_seen < crossings[0]):
                                    last_time_seen = crossings[0]
                                    after_midnight = True

                            # evening twilight
                            target_altaz = self.get_target_altaz(crossings[1])
                            if target_altaz.alt.deg > altitude_threshold:
                                month_num = month
                                if first_time_seen is None:
                                    first_time_seen = crossings[1]
                                    before_midnight = True
                                if last_time_seen is None or (last_time_seen is not None and last_time_seen < crossings[1]):
                                    last_time_seen = crossings[1]
                                    before_midnight = True

                            # evening twilight to midnight
                            for hour in self.hourly_iterator(crossings[1], crossings[1].replace(hour=23, minute=59, second=59, microsecond=0)):
                                target_altaz = self.get_target_altaz(hour)
                                if target_altaz.alt.deg > altitude_threshold:
                                    month_num = month
                                    if first_time_seen is None:
                                        first_time_seen = hour
                                        before_midnight = True
                                    if last_time_seen is None or (last_time_seen is not None and last_time_seen < hour):
                                        last_time_seen = hour
                                        before_midnight = True

                            #morning = crossings[0] + datetime.timedelta(days=1)

                except Exception:
                    continue
            if month_num is not None:
                months.append({
                    "month": month_num,
                    "first_time_seen": datetime.datetime.strftime(first_time_seen,"%m/%d"),
                    "after_midnight": after_midnight,
                    "last_time_seen": datetime.datetime.strftime(last_time_seen,"%m/%d"),
                    "before_midnight": before_midnight
                })
        return months

    def calculate_all_events(self) -> dict:
        """
        Calculate all three astronomical events.

        Returns:
            Dictionary with results for all three events
        """
        results = {
            'eastern_horizon_30deg': self.find_horizon_crossing_months(30),
            'meridian_at_midnight': self.find_meridian_at_midnight_months()
        }

        return results


def month_names(month_numbers: List[int]) -> List[str]:
    """Convert month numbers to month names."""
    month_map = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    return [month_map[m] for m in month_numbers]


def main():
    """Main function to run the astronomical visibility calculator."""
    parser = argparse.ArgumentParser(description='Calculate astronomical visibility events')
    parser.add_argument('--ra', type=float, required=False,
                        help='Right Ascension in degrees')
    parser.add_argument('--dec', type=float, required=False,
                        help='Declination in degrees')
    parser.add_argument('--target', type=str, required=False,
                        help='Celestial object name')
    parser.add_argument('--lat', type=float, required=True,
                        help='Observer latitude in degrees')
    parser.add_argument('--lon', type=float, required=True,
                        help='Observer longitude in degrees')

    args = parser.parse_args()

    # Create calculator instance
    calc = AstronomicalVisibilityCalculator(args.ra, args.dec, args.target, args.lat, args.lon)

    print(f"Astronomical Visibility Calculator")
    print(f"Target: RA={args.ra}°, Dec={args.dec}°")
    print(f"Object Name: {args.target}")
    print(f"Observer: Lat={args.lat}°, Lon={args.lon}°")
    if calc.target is None:
        print(f"\nNot able to find target.")
        print("=" * 60)
    else:
        print("=" * 60)

        # Calculate all events
        results = calc.calculate_all_events()

        # Display results
        print(
            "\n1. Months when object is 30° above Eastern horizon after astronomical twilight starts or 30° above Western horizon before astronomical twilight ends:")
        eastern_months = results['eastern_horizon_30deg']
 
        if eastern_months:
            for result in eastern_months:
                month_name = month_names([result["month"]])[0]
                when_visible = " after astronomical twilight starts"
                if result['after_midnight'] and result['before_midnight']:
                    when_visible = " after astronomical twilight starts and before astronomical twilight ends"
                elif result['after_midnight']:
                    when_visible = " before astronomical twilight ends"
                print(f"   {month_name}: earliest {result['first_time_seen']}{when_visible} / latest {result['last_time_seen']}{when_visible}")
        else:
            print("   Never occurs (object may be too far south or other constraints)")

        print("\n2. Month and day when object is at the meridian at midnight:")
        meridian_month = month_names([results['meridian_at_midnight']["month"]])
        if meridian_month:
            print("   " + meridian_month[0] + ", " + str(results['meridian_at_midnight']["day"]))
        else:
            print("   Never occurs (object may not pass through meridian at this latitude)")

        print("\nNote: Results are approximate and based on 2024 calculations.")
        print("Actual dates may vary slightly due to Earth's orbital variations.")

if __name__ == "__main__":
    main()
