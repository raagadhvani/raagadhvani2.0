import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
#importing required libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import jenkspy
import json
import streamlit as st
import numpy as np
import wave
import struct

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

############################## Initialize ##################################


# Some Useful Variables
window_size = 2205    # Size of window to be used for detecting silence
beta = 1   # Silence detection parameter
max_notes = 10000    # Maximum number of notes in file, for efficiency
sampling_freq = 44100	# Sampling frequency of audio signal
threshold = 1600


#C C# D D# E F F# G G# A A# B
#Sa Ri1 Ri2/Ga1 Ri3/Ga2 Ga3 Ma1 Ma2 Pa Dha1 Dha2/Ni1 Dha3/Ni2 Ni3


# Array for musical notes and their notations

# C C# D D# E F F# G G# A A# B
# Sa Ri1 Ri2/Ga1 Ri3/Ga2 Ga3 Ma1 Ma2 Pa Dha1 Dha2/Ni1 Dha3/Ni2 Ni3

notes = [
    'S_0', 'R1_0', 'R2_0', 'G2_0', 'G3_0', 'M1_0', 'M2_0', 'P_0', 'D1_0', 'D2_0', 'N2_0', 'N3_0',
    'S_1', 'R1_1', 'R2_1', 'G2_1', 'G3_1', 'M1_1', 'M2_1', 'P_1', 'D1_1', 'D2_1', 'N2_1', 'N3_1',
    'S_2', 'R1_2', 'R2_2', 'G2_2', 'G3_2', 'M1_2', 'M2_2', 'P_2', 'D1_2', 'D2_2', 'N2_2', 'N3_2',
    'S_3', 'R1_3', 'R2_3', 'G2_3', 'G3_3', 'M1_3', 'M2_3', 'P_3', 'D1_3', 'D2_3', 'N2_3', 'N3_3',
    'S_4', 'R1_4', 'R2_4', 'G2_4', 'G3_4', 'M1_4', 'M2_4', 'P_4', 'D1_4', 'D2_4', 'N2_4', 'N3_4',
    'S_5', 'R1_5', 'R2_5', 'G2_5', 'G3_5', 'M1_5', 'M2_5', 'P_5', 'D1_5', 'D2_5', 'N2_5', 'N3_5',
    'S_6', 'R1_6', 'R2_6', 'G2_6', 'G3_6', 'M1_6', 'M2_6', 'P_6', 'D1_6', 'D2_6', 'N2_6', 'N3_6',
    'S_7', 'R1_7', 'R2_7', 'G2_7', 'G3_7', 'M1_7', 'M2_7', 'P_7', 'D1_7', 'D2_7', 'N2_7', 'N3_7',
    'S_8', 'R1_8', 'R2_8', 'G2_8', 'G3_8', 'M1_8', 'M2_8', 'P_8', 'D1_8', 'D2_8', 'N2_8', 'N3_8'
]


# Array for corresponding frequencies in Hertz
array = [
    16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87,
    32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74,
    65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
    1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
    2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07,
    4186.01,4434.92,4698.63,4978.03,5274.04,5587.65,5919.91,6271.93,6644.88,7040.00,7458.62,7902.13
      
]
Identified_Notes = []


#title of page
st.title("Raagdhvani 🎵")
st.header("Classify your song")                                                                                                                                                  

#disabling warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#hiding menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


from io import StringIO

uploaded_file = st.file_uploader("Choose a file to classify (wav/mp3)")

if uploaded_file is not None:
    ############################## Read Audio File #############################
    st.write('\n\nReading Audio File...')

    sound_file = wave.open(uploaded_file, 'r')
    file_length = sound_file.getnframes()
    print(sound_file.getparams())
    print(file_length)
    sound = np.zeros(file_length)
    mean_square = []
    sound_square = np.zeros(file_length)
    for i in range(file_length):
        data = sound_file.readframes(1)
        data = struct.unpack("hh", data)
        sound[i] = int(data[0])

    #print(sound)
    sound = np.divide(sound, float(2**15))	# Normalize data in range -1 to 1
    #st.write("after normlaizing\n")
    #st.write(sound)
    ######################### DETECTING SCILENCE ##################################

    sound_square = np.square(sound)
    frequency = []
    dft = []
    i = 0
    j = 0
    k = 0    
    # traversing sound_square array with a fixed window_size
    while(i<=len(sound_square)-window_size):
        s = 0.0
        j = 0
        while(j<=window_size):
            s = s + sound_square[i+j]
            j = j + 1	
    # detecting the silence waves
        if s < threshold:
            if(i-k>window_size*4):
                dft = np.array(dft) # applying fourier transform function
                dft = np.fft.fft(sound[k:i])
                dft=np.argsort(dft)

                if(dft[0]>dft[-1] and dft[1]>dft[-1]):
                    i_max = dft[-1]
                elif(dft[1]>dft[0] and dft[-1]>dft[0]):
                    i_max = dft[0]
                else :	
                    i_max = dft[1]
    # claculating frequency				
                frequency.append((i_max*sampling_freq)/(i-k))
                dft = []
                k = i+1
        i = i + window_size

     

    for i in frequency :
        print(i)
        idx = (np.abs(array-i)).argmin()
        Identified_Notes.append(notes[idx])
    #st.write(Identified_Notes)



    # Convert the output array to a comma-separated string
    output_string = ', '.join(Identified_Notes)

    # Print the result
    st.write("Identified Notes \n")
    st.write(output_string)

    def reduce_consecutive_duplicates(note_list):
        reduced_list = [note_list[0]]  # Initialize the reduced list with the first note
        for i in range(1, len(note_list)):
            if note_list[i] != note_list[i - 1]:  # Check if the current note is different from the previous note
                reduced_list.append(note_list[i])  # If different, add it to the reduced list
        return reduced_list

    
    reduced_notes=reduce_consecutive_duplicates(Identified_Notes)


    # Convert the output array to a comma-separated string
    output_string = ', '.join(reduced_notes)

    # Print the result
    st.write("Reduced Notes \n")
    st.write(output_string)

    import matplotlib.pyplot as plt

    output_array = Identified_Notes

    possible_notes = ['S', 'R1', 'R2', 'G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2', 'N2', 'N3']

    note_counts = {note: 0 for note in possible_notes}

    for item in output_array:
        note = item.split('_')[0]
        if note in possible_notes:
            note_counts[note] += 1

    # Extract note names and their corresponding counts
    notes = list(note_counts.keys())
    counts = list(note_counts.values())

    # Create a Streamlit app
    st.write("Frequency of Occurrence of Each Note")

    # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(notes, counts)

    # Add labels
    ax.set_xlabel("Notes")
    ax.set_ylabel("Frequency of Occurrence")

    # Display the chart in Streamlit
    st.pyplot(fig)



    possible_notes = ['S', 'R1', 'R2', 'G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2', 'N2', 'N3']

    # Define the notes for Bhavapriya and Shankarabharana ragas
    bhavapriya_notes = ['S', 'R1', 'G2', 'M2', 'P', 'D1', 'N2']
    shankarabharana_notes = ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N2']

    note_counts = {note: 0 for note in possible_notes}

    for item in output_array:
        note = item.split('_')[0]
        if note in possible_notes:
            note_counts[note] += 1

    # Calculate the total count of notes
    total_count = sum(note_counts.values())

    # Calculate the percentages
    percentages = {note: (count / total_count) * 100 for note, count in note_counts.items()}

    # Calculate the percentages for Bhavapriya and Shankarabharana
    bhavapriya_percentage = sum(percentages[note] for note in bhavapriya_notes)
    shankarabharana_percentage = sum(percentages[note] for note in shankarabharana_notes)

    # Create a Streamlit app
    st.title("Percentage of Occurrence for Bhavapriya and Shankarabharana")

    # Display the percentages for each raga
    st.write("Percentage of Bhavapriya:", bhavapriya_percentage)
    st.write("Percentage of Shankarabharana:", shankarabharana_percentage)

    # Create a pie chart to visualize the percentages
    fig, ax = plt.subplots()
    labels = ['Bhavapriya', 'Shankarabharana']
    sizes = [bhavapriya_percentage, shankarabharana_percentage]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    # Display the pie chart in Streamlit
    st.pyplot(fig)


    





    output_array = Identified_Notes
    # Create a dictionary to count the occurrences of each note
    note_counts = {}

    for item in output_array:
        note, _ = item.split('_')
        if note not in note_counts:
            note_counts[note] = 0
        note_counts[note] += 1

    # Define the note categories to consider
    note_categories = {
        'S': ['S'],
        'R': ['R1', 'R2'],
        'G': ['G2', 'G3'],
        'M': ['M1', 'M2'],
        'P': ['P'],
        'D': ['D1', 'D2'],
        'N': ['N2', 'N3']
    }

    # Initialize a dictionary to store the most occurring notes for each category
    most_occuring_notes = {}

    # Iterate through note_categories and find the most occurring note for each category
    for category, notes in note_categories.items():
        max_count = 0
        most_occuring_note = None
        for note in notes:
            if note in note_counts and note_counts[note] > max_count:
                max_count = note_counts[note]
                most_occuring_note = note
        if most_occuring_note:
            most_occuring_notes[category] = most_occuring_note

    





























    import streamlit as st

    # Given output
    output_array = Identified_Notes

    # Define the notes for Shankarabharana and Bhavapriya ragas
    shankarabharana_notes = ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3']
    bhavapriya_notes = ['S', 'R1', 'G2', 'M2', 'P', 'D1', 'N2']

    # Count the occurrences of each note in the output
    shankarabharana_count = {note: 0 for note in shankarabharana_notes}
    bhavapriya_count = {note: 0 for note in bhavapriya_notes}

    for item in output_array:
        note, octave = item.split('_')
        if note in shankarabharana_count:
            shankarabharana_count[note] += 1
        if note in bhavapriya_count:
            bhavapriya_count[note] += 1

    # Calculate the percentage likelihood for Shankarabharana and Bhavapriya ragas
    total_notes = len(output_array)
    shankarabharana_likelihood = sum(shankarabharana_count.values()) / total_notes
    bhavapriya_likelihood = sum(bhavapriya_count.values()) / total_notes

    # Create a Streamlit app
    st.title("Raga Detection")

    # Display the likelihood as percentages
    st.write("Shankarabharana Likelihood:", f"{shankarabharana_likelihood:.2%}")
    st.write("Bhavapriya Likelihood:", f"{bhavapriya_likelihood:.2%}")

    # Visualize the likelihood as a bar chart
    st.bar_chart({
        "Shankarabharana": shankarabharana_likelihood,
        "Bhavapriya": bhavapriya_likelihood
    })




    st.title("Most occuring Notes for each Swara category")
        # Define the raga based on the most occurring notes
    raga = None
    if all(note in most_occuring_notes.values() for note in shankarabharana_notes):
        raga = "Shankarabharana"
    elif all(note in most_occuring_notes.values() for note in bhavapriya_notes):
        raga = "Bhavapriya"

    # Print the most occurring notes for each category and the matched raga
    for category, note in most_occuring_notes.items():
        st.write(f"Most occurring {category} note: {note}")

    st.write(f"Matched Raga: {raga}")




       