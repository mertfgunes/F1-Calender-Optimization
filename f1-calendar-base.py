import unittest
import math
import csv
import random
import copy
import numpy as np
import pyswarms as ps
from simanneal import Annealer
from deap import base, creator, tools

#Progress checking:
#14:06 12/12/2024
#tests are finished
#133742.2415190981 is the best result

# the unit tests to check that the simulation has been implemented correctly
# class UnitTests (unittest.TestCase):
#     # this will read in the track locations file and will pick out 5 fields to see if the file has been read correctly
#     def testReadCSV(self):
#         # read in the locations file
#         rows = readCSVFile('track-locations.csv')
#
#         # test that the corners and a middle value are read in correctly
#         self.assertEqual('GP', rows[0][0])
#         self.assertEqual('Abu Dhabi', rows[0][24])
#         self.assertEqual('temp week 52', rows[55][0])
#         self.assertEqual('24.4', rows[55][24])
#         self.assertEqual('14.5', rows[11][8])
#
#     # this will test to see if the row conversion works. here we will convert the latitude rwo and will test 5 values
#     # as we are dealing with floating point we will use almost equals rather than a direct equality
#     def testRowToFloat(self):
#         # read in the locations file and convert the latitude column to floats
#         rows = readCSVFile('track-locations.csv')
#         convertRowToFloat(rows, 2)
#
#         # check that 5 of the values have converted correctly
#         self.assertAlmostEqual(26.0325, rows[2][1], delta=0.0001)
#         self.assertAlmostEqual(24.4672, rows[2][24], delta=0.0001)
#         self.assertAlmostEqual(40.3725, rows[2][17], delta=0.0001)
#         self.assertAlmostEqual(30.1327, rows[2][19], delta=0.0001)
#         self.assertAlmostEqual(25.49, rows[2][23], delta=0.0001)
#
#         # check that the conversion of a temperature row to floating point is also correct
#         convertRowToFloat(rows, 5)
#
#         # check that 5 of the values have converted correctly
#         self.assertAlmostEqual(20.5, rows[5][1], delta=0.0001)
#         self.assertAlmostEqual(24.5, rows[5][24], delta=0.0001)
#         self.assertAlmostEqual(7.25, rows[5][17], delta=0.0001)
#         self.assertAlmostEqual(16.75, rows[5][19], delta=0.0001)
#         self.assertAlmostEqual(22.5, rows[5][23], delta=0.0001)
#
#     # this will test to see if the file conversion overall is successful for the track locations
#     # it will read in the file and will test a string, float, and int from 2 rows to verify it worked correctly
#     def testReadTrackLocations(self):
#         # read in the locations file
#         rows = readTrackLocations()
#
#         # check the name, latitude, and final temp of the first race
#         self.assertEqual(rows[0][0], 'Bahrain')
#         self.assertAlmostEqual(rows[2][0], 26.0325, delta=0.0001)
#         self.assertAlmostEqual(rows[55][0], 20.6, delta=0.0001)
#
#         # check the name, longitude, and initial temp of the last race
#         self.assertEqual(rows[0][23], 'Abu Dhabi')
#         self.assertAlmostEqual(rows[2][23], 24.4672, delta=0.0001)
#         self.assertAlmostEqual(rows[4][23], 24, delta=0.0001)
#
#     # tests to see if the race weekends file is read in correctly
#     def testReadRaceWeekends(self):
#         # read in the race weekends file
#         weekends = readRaceWeekends()
#
#         # check that bahrain is weekend 9 and abu dhabi is weekend 49
#         self.assertEqual(weekends[0], 9)
#         self.assertEqual(weekends[23], 49)
#
#         # check that hungaroring is weekend 29
#         self.assertEqual(weekends[12], 29)
#
#     # this will test to see if the haversine function will work correctly we will test 4 sets of locations
#     def testHaversine(self):
#         # read in the locations file with conversion
#         rows = readTrackLocations()
#
#         # check the distance of Bahrain against itself this should be zero
#         self.assertAlmostEqual(haversine(rows, 0, 0), 0.0, delta=0.01)
#
#         # check the distance of Bahrain against Silverstone this should be 5158.08 km
#         self.assertAlmostEqual(haversine(rows, 0, 11), 5158.08, delta=0.01)
#
#         # check the distance of silverstone against monza this should be 1039.49 Km
#         self.assertAlmostEqual(haversine(rows, 11, 15), 1039.49, delta=0.01)
#
#         # check the distance of monza to the red bull ring this should be 455.69 Km
#         self.assertAlmostEqual(haversine(rows, 10, 15), 455.69, delta=0.01)
#
#     # will test to see if the season distance calculation is correct using the 2023 calendar
#     def testDistanceCalculation(self):
#         # read in the locations & race weekends, generate the weekends, and calculate the season distance
#         tracks = readTrackLocations()
#         weekends = readRaceWeekends()
#
#         # calculate the season distance using silverstone as the home track as this will be the case for 8 of the teams we will use monza
#         # for the other two teams.
#         self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 11), 193645.3219, delta=0.0001)
#         self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 15), 191408.7483, delta=0.0001)
#
#     # will test that the temperature constraint is working this should fail as azerbijan should fail the test
#     def testTempConstraint(self):
#         # load in the tracks, race weekends, and the sundays
#         tracks = readTrackLocations()
#         weekends1 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 34, 35, 37, 38, 42, 43, 44, 47, 48, 49]
#         weekends2 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 31, 35, 37, 38, 42, 43, 44, 40, 48, 49]
#
#         # the test with the default calender should be false because of zaandvort
#         self.assertEqual(checkTemperatureConstraint(tracks, weekends1, 20, 35), False)
#         self.assertEqual(checkTemperatureConstraint(tracks, weekends2, 20, 35), True)
#
#     # will test that we can detect four race weekends in a row.
#     def testFourRaceInRow(self):
#         # weekend patterns the first does not have four in a row the second does
#         weekends1 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 34, 35, 37, 38, 42, 43, 44, 47, 48, 49]
#         weekends2 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 35, 36, 37, 38, 42, 43, 44, 47, 48, 49]
#
#         # the first should pass and the second should fail
#         self.assertEqual(checkFourRaceInRow(weekends1), False)
#         self.assertEqual(checkFourRaceInRow(weekends2), True)
#
#     # will test that we can detect a period for a summer shutdown in the prescribed weeks
#     def testSummerShutdown(self):
#         # weekend patterns the first has a summer shutdown the second doesn't
#         weekends1 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 34, 35, 37, 38, 42, 43, 44, 47, 48, 49]
#         weekends2 = [9, 10, 12, 14, 16, 18, 20, 21, 23, 25, 26, 27, 29, 30, 33, 35, 37, 38, 42, 43, 44, 47, 48, 49]
#
#         # the first should pass and the second should fail
#         self.assertEqual(checkSummerShutdown(weekends1), True)
#         self.assertEqual(checkSummerShutdown(weekends2), False)
#
# function that will calculate the total distance for the season assuming a given racetrack as the home racetrack
# the following will be assumed:
# - on a weekend where there is no race the team will return home
# - on a weekend in a double or triple header a team will travel straight to the next race and won't go back home
# - the preseason test will always take place in Bahrain
# - for the summer shutdown and off season the team will return home
def calculateSeasonDistance(tracks, weekends, home):
    total_distance = 0
    current_location = home  # first the home track

    for i in range(1, 53):  # iteration through entire year
        if i in weekends:
            # race weekend -> find the index of the race track
            race_index = weekends.index(i)

            # calculation of distance from current location to next
            total_distance += haversine(tracks, current_location, race_index)

            # update the location
            current_location = race_index

            # check for next week, whether its a race or not.
            if i == weekends[-1] or (i + 1 not in weekends):
                total_distance += haversine(tracks, current_location, home)
                current_location = home

    # end of the season = home
    if current_location != home:
        total_distance += haversine(tracks, current_location, home)

    return total_distance

# function that will check to see if there is anywhere in our weekends where four races appear in a row. True indicates that we have four in a row
def checkFourRaceInRow(weekends):
    # sor the weekends
    weekends = sorted(weekends)

    # checker
    for i in range(len(weekends) - 3):
        if weekends[i] + 1 == weekends[i + 1] and \
                weekends[i] + 2 == weekends[i + 2] and \
                weekends[i] + 3 == weekends[i + 3]:
            return True

    return False

# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of min degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, min, max):
    i= 0

    for j in range (23):
        if tracks[weekends[i] + 4][j] < min or tracks[weekends[i] + 4][j] > max:
            return False
        else:
            i += 1
    return True
# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races. this will be weeks 31, 32, and 33, they are not
# permitted to have a race during these weekends
def checkSummerShutdown(weekends):


    #checks if there are any three repeating weekends with no race until summer


    # range define
    summer_weeks = set(range(27, 36))

    # checking all the race weekends
    race_weekends = set(weekends)

    # if the non race weekend in the summer
    non_race_weekends = sorted([week for week in summer_weeks if week not in race_weekends])

    # is there 3 repeating race?
    for i in range(len(non_race_weekends) - 2):
        if non_race_weekends[i] + 1 == non_race_weekends[i + 1] and non_race_weekends[i] + 2 == non_race_weekends[i + 2]:
            return True

    return False

# will go through the genetic code of this child and will make sure that all the required weekends are in it.
# it's highly likely that with crossover that there will be weekends missing and others duplicated. we will
# randomly replace the duplicated ones with the missing ones
def childGeneticCodeFix(child):
    pass

# function that will take in the set of rows and will convert the given row index into floating point values
# this assumes the header in the CSV file is still present so it will skip the first column
def convertRowToFloat(rows, row_index):
    """
    converts all elements in a specific row of the rows list to floating-point numbers where possible
    """
    for col_index in range(1, len(rows[row_index])):  # avoid header
        try:
            rows[row_index][col_index] = float(rows[row_index][col_index])
        except ValueError:
            pass


# function that will generate a shuffled itinerary. However, this will make sure that the bahrain, abu dhabi, and monaco
# will retain their fixed weeks in the calendar
def generateShuffledItinerary(weekends):
    pass

# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.
def haversine(rows, location1, location2):
    # extract both in degrees
    lat1 = rows[2][location1]
    lon1 = rows[3][location1]
    lat2 = rows[2][location2]
    lon2 = rows[3][location2]

    # degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # the haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = 6371 * c  # radius of earth

    return distance

# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexHighestTemp(tracks, weekends, max):
    pass

# function that will give us the index of the lowest temp below min. will return -1 if none found
def indexLowestTemp(tracks, weekends, min):
    pass

# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home):
    print(f"--- Preseason Test at {home} ---\n")

    # Go through each race weekend
    for i, (track, weekend) in enumerate(zip(tracks, weekends), start=1):
        print(f"Weekend {i}: {weekend}")
        print(f"   Travel to {track}")
        print(f"   Race Weekend at {track}")
        print(f"   Return to {home}\n")

    print("--- Itinerary Complete ---")

# function that will take in the given CSV file and will read in its entire contents
# and return a list of lists
def readCSVFile(file):
    # the rows to return
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open(file)
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows.
    for row in csv_reader:
        rows.append(row)

    # close the file when reading is finished
    csv_file.close()

    # return the rows at the end of the function
    return rows

# function that will read in the race weekends file and will perform all necessary conversions on it
def readRaceWeekends():

    # read
    rows = readCSVFile('race-weekends.csv')

    # convert the second row into integer, skip the header
    weekends = [int(row[1]) for row in rows[1:]]

    return weekends

# function that will read the track locations file and will perform all necessary conversions on it.
# this should also strip out the first column on the left which is the header information for all the rows
def readTrackLocations():
    # read
    rows = readCSVFile('track-locations.csv')

    # convert row 2 and row 3 to floats
    convertRowToFloat(rows, 2)  # latitude
    convertRowToFloat(rows, 3)  # longitude

    # convert all temperature rows to floats
    for i in range(4, len(rows)):
        convertRowToFloat(rows, i)

    # return first column without the header.
    return [row[1:] for row in rows]

# function that performs a roulette wheel randomisation on the two given values and returns the chosen on
def rouletteWheel(a, b):

    return a if random.random() < 0.5 else b


class CalendarAnnealer(Annealer):
    def __init__(self, state, tracks, weekends, home):
        super().__init__(state)
        self.tracks = tracks
        self.home = home
        self.original_weekends = weekends.copy()

    def move(self):
        """
        Perform a prioritized move to resolve constraint violations.
        """
        # Identify indices of races violating constraints
        invalid_temp_indices = [
            i for i, weekend in enumerate(self.state)
            if self.tracks[weekend][4] < 15 or self.tracks[weekend][4] > 35
        ]

        if invalid_temp_indices:
            # Swap one violating race with a valid one
            idx1 = random.choice(invalid_temp_indices)
            idx2 = random.choice([i for i in range(len(self.state)) if i not in [8, 20, 48] and i != idx1])
            self.state[idx1], self.state[idx2] = self.state[idx2], self.state[idx1]
        else:
            # Fallback: random swap
            idx1, idx2 = random.sample([i for i in range(len(self.state)) if i not in [8, 20, 48]], 2)
            self.state[idx1], self.state[idx2] = self.state[idx2], self.state[idx1]

    def energy(self):
        weekends = self.state
        total_distance = calculateSeasonDistance(self.tracks, weekends, self.home)

        # Penalty for temperature constraints
        temp_penalty = 0
        for weekend in weekends:
            temp = self.tracks[weekend][4]
            if temp < 15:
                temp_penalty += (15 - temp) ** 2  # Squared penalty for being too cold
            elif temp > 35:
                temp_penalty += (temp - 35) ** 2  # Squared penalty for being too hot

        # Penalty for summer shutdown violation
        shutdown_penalty = 100000 if not checkSummerShutdown(weekends) else 0

        # Penalty for four consecutive races
        four_races_penalty = 100000 if checkFourRaceInRow(weekends) else 0

        return total_distance + temp_penalty + shutdown_penalty + four_races_penalty

    def generateShuffledItinerary(weekends):
        """
        Shuffle the weekends while keeping Bahrain, Monaco, and Abu Dhabi fixed.
        """
        fixed_indices = [8, 20, 48]
        fixed_values = [weekends[i] for i in fixed_indices]

        # Extract the non-fixed elements
        non_fixed_indices = [i for i in range(len(weekends)) if i not in fixed_indices]
        non_fixed_values = [weekends[i] for i in non_fixed_indices]

        # Shuffle the non-fixed values
        random.shuffle(non_fixed_values)

        # Merge back the fixed and shuffled non-fixed values
        shuffled_weekends = weekends[:]
        for idx, value in zip(non_fixed_indices, non_fixed_values):
            shuffled_weekends[idx] = value

        return shuffled_weekends

    def printItinerary(tracks, weekends, home):
        """
        Prints the itinerary for the season in a readable format.
        """
        print(f"--- Preseason Test at Home Track ({tracks[0][home]}) ---\n")
        total_distance = 0

        for i, weekend in enumerate(weekends, start=1):
            track_name = tracks[0][weekend]  # Assuming track names are in the first row
            expected_temp = tracks[4][weekend]  # Assuming temperature row is index 4

            print(f"Weekend {i}: {track_name} (Week {weekend})")
            print(f"   - Expected Temperature: {expected_temp:.1f}°C")

            if i == 1:
                print(f"   - Travel from Home to {track_name}")
                distance = haversine(tracks, home, weekend)
            elif i == len(weekends):
                print(f"   - Travel to {track_name} and return to Home")
                distance = haversine(tracks, weekends[i - 2], weekend) + haversine(tracks, weekend, home)
            else:
                print(f"   - Travel from Previous Race to {track_name}")
                distance = haversine(tracks, weekends[i - 2], weekend)

            total_distance += distance
            print(f"   - Distance Traveled: {distance:.2f} km\n")

        print(f"--- Total Distance for the Season: {total_distance:.2f} km ---\n")

    def validateItinerary(tracks, weekends):
        """
        Validates the generated itinerary and prints results.
        """
        temp_valid = checkTemperatureConstraint(tracks, weekends, 15, 35)
        four_race_valid = not checkFourRaceInRow(weekends)
        summer_valid = checkSummerShutdown(weekends)

        print("--- Itinerary Validation ---")
        print(f" - Temperature Constraint: {'Passed' if temp_valid else 'Failed'}")
        print(f" - Consecutive Races Constraint: {'Passed' if four_race_valid else 'Failed'}")
        print(f" - Summer Shutdown Constraint: {'Passed' if summer_valid else 'Failed'}")
        print(f"--- Validation Complete ---\n")


def SAcases():
    """
    Runs the Simulated Annealing case and prints results for both Silverstone and Monza.
    """
    # Read tracks and weekends
    tracks = readTrackLocations()
    weekends = readRaceWeekends()

    # Initial state
    initial_state = weekends.copy()

    # Annealing for Silverstone
    print("--- Simulated Annealing: Silverstone ---")
    annealer = CalendarAnnealer(initial_state, tracks, weekends, 11)
    final_state, final_energy = annealer.anneal()
    print("Final Distance (Silverstone):", final_energy)
    printItinerary(tracks, final_state, 11)

    # Annealing for Monza
    print("--- Simulated Annealing: Monza ---")
    annealer = CalendarAnnealer(initial_state, tracks, weekends, 15)
    final_state, final_energy = annealer.anneal()
    print("Final Distance (Monza):", final_energy)
    printItinerary(tracks, final_state, 15)


# function that will run the genetic algorithms cases for all four situations
def GAcases():
    pass

# function that will run particle swarm optimisation in an attempt to find a solution
def PSOcases():
    pass

if __name__ == '__main__':
    # uncomment this run all the unit tests. when you have satisfied all the unit tests you will have a working simulation
    # you can then comment this out and move onto your SA and GA solutions
    # unittest.main()

    # just to check that the itinerary printing mechanism works. we will assume that silverstone is the home track for this
    weekends = readRaceWeekends()
    print(generateShuffledItinerary(weekends))
    tracks = readTrackLocations()
    printItinerary(tracks, weekends, 11)

    # run the cases for simulated annealing
    SAcases()

    # run the cases for genetic algorithms
    #GAcases()

    # run the cases for particle swarm optimisation
    #PSOcases()